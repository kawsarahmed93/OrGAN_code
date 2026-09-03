from argparse import ArgumentParser

import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import time
import torch.nn.functional as F
from PIL import Image

from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from datasets import NIH_IMG_BOX_LEVEL_DS, get_bbox_test_transforms, collate_fn_img_level_ds
from models import DenseNet121, FusionNet
from configs import all_configs, DEVICE, NIH_DATASET_ROOT_DIR, LABEL_DIR, NIH_CXR_SINGLE_LABEL_NAMES
from trainer_callbacks import set_random_state
import matplotlib.pyplot as plt
import cv2

# Prevent each DataLoader worker process from also spawning its own OpenCV
# thread pool - with num_workers>0 that oversubscribes CPUs and can make
# preprocessing slower, not faster, on a shared node.
cv2.setNumThreads(0)

DISEASE_ID_TO_NAME = {
    0: 'Atelectasis', 
    1: 'Cardiomegaly', 
    2: 'Effusion', 
    3: 'Infiltrate', 
    4: 'Mass', 
    5: 'Nodule', 
    6: 'Pneumonia', 
    7: 'Pneumothorax'
    }

DISEASE_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']


def save_with_retry(save_func, path, retries=5, delay=2.0):
    """Runs save_func(path), retrying on transient network-filesystem errors.

    The shared storage this repo runs on intermittently reports an existing
    directory as missing to a subsequent stat/open ('Remote I/O error' /
    'Parent directory ... does not exist'), even immediately after a
    successful write to that same directory. pandas' to_excel is especially
    exposed because it stats the parent via check_parent_directory() before
    opening the file, so a single bad stat aborts the whole run after the
    work is already done. Retrying after a short delay, re-creating the
    directory each time, works around that without masking a genuine
    missing-path bug (which would still fail after all retries).

    Mirrors the helper of the same name in the OrGAN train-* scripts.
    """
    parent_dir = os.path.dirname(path)
    last_err = None
    for attempt in range(retries):
        try:
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            save_func(path)
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                print(f'Save to {path} failed ({e}); retrying ({attempt + 1}/{retries})...')
                time.sleep(delay)
    raise last_err


def get_args():
    parser = ArgumentParser(description='test')
    # parser.add_argument('--run_configs_list', type=str, default=['xray_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['lung_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['bone_base'])
    parser.add_argument('--run_configs_list', type=str, nargs="*", default=['segment_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['proposed'])
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--image_resize_dim', type=int, default=586)
    # parser.add_argument('--image_crop_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--images-root', dest='images_root', type=str, default=None,
                        help='Override the image directory (must end in /). Point this at a '
                             'node-local copy made by stage_bbox_images.sh when the CBIG NFS '
                             'export is slow - it is read once per fold, so 5x the reads.')
    args = parser.parse_args()
    return args

class GradCAMPlusPlus:
    def __init__(self, model, target_layer, method):
        self.model = model
        self.target_layer = target_layer
        self.method = method

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, l, class_idx):
        self.model.zero_grad()
        x = x.requires_grad_(True)
        l = l.requires_grad_(True)

        if self.method=='base':
            output = self.model(x)
        elif self.method=='proposed':
            output = self.model(x,l)
            
        logits = output["logits"]

        if isinstance(class_idx, torch.Tensor):
            class_idx = class_idx.item()
        
        score = logits[:, class_idx].sum()

        # Backward pass
        score.backward(retain_graph=True)

        grads = self.gradients          # (B, C, H, W)
        acts = self.activations         # (B, C, H, W)

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        # Sum over spatial dims
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        eps = 1e-8

        # α coefficients (pixel-wise)
        alpha = grads_power_2 / (
            2 * grads_power_2 + sum_acts * grads_power_3 + eps
        )

        # Positive gradients only
        relu_grads = F.relu(grads)

        # Weights
        weights = (alpha * relu_grads).sum(dim=(2, 3), keepdim=True)

        # CAM
        cam = (weights * acts).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

def main():
    args = get_args()
    kfolds=5
    
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)
            
    # get configs
    run_config = args.run_configs_list[0]  
    configs = all_configs[run_config]
    weight_saving_path = configs['weight_saving_path']
    
    set_random_state(args.seed)
    
    # get dataset
    data = np.load("./label_info/bbox_data.npy", allow_pickle=True).item()
    
    images_root = args.images_root or NIH_DATASET_ROOT_DIR
    if not images_root.endswith('/'):
        images_root += '/'
    if images_root != NIH_DATASET_ROOT_DIR:
        print(f'Reading images from override path: {images_root}')
    bbox_fpaths = np.array([images_root+x for x in data['bbox_fnames']])
    bbox_labels = np.array(data['bbox_labels'])
    bbox = data['bbox']

    # get dataloader
    print('Loading Baseline dataloader!')
    test_dataset = NIH_IMG_BOX_LEVEL_DS(
        bbox_fpaths,
        bbox_labels,
        bbox,
        configs['flag'],
        get_bbox_test_transforms(configs['resize_crop'][0], configs['resize_crop'][1]),
    )

    # GradCAM needs one sample at a time (per-sample backward + bbox match),
    # so the model-side batch size must stay 1. That doesn't mean the run has
    # to be serial though: without a DataLoader, image read + resize +
    # albumentations transform for every sample ran synchronously on the
    # main process between GPU calls, starving the GPU. Wrapping the same
    # dataset in a DataLoader with num_workers lets those CPU-bound steps
    # prefetch in parallel while the GPU is busy with the previous sample.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        persistent_workers=args.n_workers > 0,
        collate_fn=collate_fn_img_level_ds,
    )

    # GLOBAL accumulators across all folds
    # Per-fold IoU accuracy, for the cross-fold mean +- s.d. that gets reported.
    # Every fold evaluates the SAME held-out bbox set with a different model, so
    # the spread across these is model variability with the test set held fixed.
    per_fold_iou_acc = []
    per_fold_micro = []
    per_fold_counts = None

    global_iou = defaultdict(list)
    global_dice = defaultdict(list)
    global_pre = defaultdict(list)
    global_re = defaultdict(list)
    global_f1 = defaultdict(list)

    # fold_scores = {
    #     "IoU": [],
    #     "Dice": [],
    #     "F1": [],
    #     "Precision": [],
    #     "Recall": []
    # }
    
    for fold in range(kfolds):
    
        if configs['model_type'] == 'densenet121':
            model = DenseNet121(args.num_classes)
        elif configs['model_type'] == 'fusion_net':
            model = FusionNet(args.num_classes)
            
        # checkpoint = torch.load(weight_saving_path + '/fold' + str(self.fold) + '/'+ f'checkpoint_best_loss_fold{fold}.pth')
        checkpoint = torch.load(weight_saving_path + 'fold' + str(fold) + '/'+ f'checkpoint_best_auc_fold{fold}.pth')
        print('loss score: {:.4f}'.format(checkpoint['val_loss']))
        print('auc score: {:.4f}'.format(checkpoint['val_auc']))
        model.load_state_dict(checkpoint['Model_state_dict'])
        model = model.to(DEVICE)
        # if len(args.gpu_ids) > 1:
        #     model = nn.DataParallel(model, device_ids=args.gpu_ids)
        model.eval()                  
        del checkpoint
    
        
        if configs['method'] == 'base':
            # target_layer = model.encoder[-2]
            target_layer = model.norm5
        elif configs['method'] == 'proposed':
            # target_layer = model.encoder.denseblock4
            target_layer = model.encoder.fusion4
            
        gradcam = GradCAMPlusPlus(
                model,
                target_layer,
                configs['method']
            )
        
        iou_results = defaultdict(list)
        dice_results = defaultdict(list)
        pre_results = defaultdict(list)
        re_results = defaultdict(list)
        f1_results = defaultdict(list)
        
        
        for data in tqdm(test_loader, total=len(test_loader)):

            images = data['image'].to(DEVICE).float()
            lungs = data['lung'].to(DEVICE).float()

            # DataLoader adds a batch dim (size 1 here) that generate()/
            # eval_model() below don't expect - strip it back off.
            targets = data['target'][0]
            ibbox = data['bbox'][0]
        
            # ----------------------------
            # IMPORTANT: NO return_logit_maps
            # ----------------------------
            cam = gradcam.generate(images, lungs, class_idx=torch.argmax(targets).item())
            # print(cam.shape)
            # ----------------------------
            # resize CAM
            # ----------------------------
            cam = cam.unsqueeze(1) # (1,1,H,W)
        
            cam = F.interpolate(
                cam,
                (configs['resize_crop'][0], configs['resize_crop'][1]),
                mode='bilinear',
                align_corners=False
            )
        
            cam = cam.squeeze().detach().cpu().numpy()
        
            # ----------------------------
            # normalize
            # ----------------------------
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
                
            cam = (cam * 255).astype(np.uint8)
        
            # ----------------------------
            # GT handling
            # ----------------------------
            # image_id = data['image_id']
            # image_root_dir = NIH_DATASET_ROOT_DIR
            # save_root_dir = './attention_maps/'
            disease_label = torch.argmax(targets).item()
        
            attention = cam  # already single-class CAM
        
            gt_box = ibbox.cpu().numpy() if torch.is_tensor(ibbox) else np.array(ibbox)
            
            metrics = eval_model(attention, gt_box, configs['pc'])
    
            iou_results[DISEASE_NAMES[disease_label]].append(metrics["IoU"])
            dice_results[DISEASE_NAMES[disease_label]].append(metrics["Dice"])
            pre_results[DISEASE_NAMES[disease_label]].append(metrics["Precision"])
            re_results[DISEASE_NAMES[disease_label]].append(metrics["Recall"])
            f1_results[DISEASE_NAMES[disease_label]].append(metrics["F1-score"])

        results=[iou_results, dice_results, pre_results, re_results, f1_results]

        # fold_summary = summarize_fold(results)

        # for k in fold_scores:
        #     fold_scores[k].append(fold_summary[k])
        
        per_fold_iou_acc.append(eval_t_iou(iou_results))
        per_fold_micro.append(eval_micro_iou_acc(iou_results))
        if per_fold_counts is None:
            # Boxes per finding for ONE fold. This is the n to report: every
            # fold sees the same test set, so global_iou below holds kfolds x
            # that many values for each class.
            per_fold_counts = {c: len(iou_results.get(c, [])) for c in DISEASE_NAMES}

        for cls in DISEASE_NAMES:
            global_iou[cls].extend(iou_results[cls])
            global_dice[cls].extend(dice_results[cls])
            global_pre[cls].extend(pre_results[cls])
            global_re[cls].extend(re_results[cls])
            global_f1[cls].extend(f1_results[cls])
        

        print(f"Fold: {fold}\n")
        
        print_iou(eval_t_iou(results[0]))
        print_results(results)

    print("\n================ FINAL CROSS-FOLD RESULTS ================\n")

    final_results = [global_iou, global_dice, global_pre, global_re, global_f1]
    final_iou_acc = eval_t_iou(global_iou)

    print_iou(final_iou_acc)
    print_results(final_results)

    metrics_df = build_metrics_df(final_results)
    # Pooled table: counts must come from a single fold, not from global_iou,
    # which holds kfolds copies of the same test set.
    pooled_iou_acc_df = build_iou_acc_df(final_iou_acc, global_iou,
                                         counts_override=per_fold_counts,
                                         micro=eval_micro_iou_acc(global_iou))
    cross_fold_iou_acc_df = build_cross_fold_iou_acc_df(per_fold_iou_acc, per_fold_counts,
                                                        per_fold_micro)

    print("\n---- IoU accuracy, mean +- s.d. across the "
          f"{len(per_fold_iou_acc)} cross-validation models ----")
    print(cross_fold_iou_acc_df.to_string(index=False))

    metrics_excel_path = weight_saving_path + f'final_cross_fold_metrics_{run_config}.xlsx'
    save_with_retry(lambda p: metrics_df.to_excel(p, index=False), metrics_excel_path)
    print(f"Saved final cross-fold metrics to {metrics_excel_path}")

    iou_acc_excel_path = weight_saving_path + f'final_iou_accuracy_{run_config}.xlsx'
    def _write_iou_sheets(path):
        with pd.ExcelWriter(path) as xl:
            cross_fold_iou_acc_df.to_excel(xl, sheet_name='cross_fold', index=False)
            pooled_iou_acc_df.to_excel(xl, sheet_name='pooled', index=False)
    save_with_retry(_write_iou_sheets, iou_acc_excel_path)
    print(f"Saved final IoU accuracy to {iou_acc_excel_path} "
          f"(sheets: cross_fold [report this], pooled)")


    # print("\n========= CROSS-FOLD RESULTS (mean ± std) =========\n")

    # for k, vals in fold_scores.items():
    #     vals = np.array(vals)
    #     print(f"{k:10s}: {vals.mean():.2f}±{vals.std():.2f}")
        

def bbox_to_mask(bbox, shape):
    """
    bbox: [x1, y1, x2, y2]
    shape: (H, W)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, bbox)
    mask[y1:y2+1, x1:x2+1] = 1
    return mask

def cam_to_mask(attention, pc):
    th = np.percentile(attention, pc)
    _, mask = cv2.threshold(attention, th, 1, cv2.THRESH_BINARY)
    # _, mask = cv2.threshold(attention, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return mask.astype(np.uint8)

def compute_confusion(pred_mask, gt_mask):
    TP = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    FP = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
    FN = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
    TN = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
    
    return TP, FP, FN, TN

def compute_metrics(pred_mask, gt_mask):
    TP, FP, FN, TN = compute_confusion(pred_mask, gt_mask)
    
    TP, FP, FN, TN = map(float, [TP, FP, FN, TN])
    eps = 1e-8
    
    iou = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2*(precision*recall)/(precision + recall + eps)
    
    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
    
def eval_model(attention, gt_box, pc=80):
    
    # Pred mask
    pred_mask = cam_to_mask(attention, pc)
    
    # GT mask
    gt_mask = bbox_to_mask(gt_box, pred_mask.shape)
    
    # Metrics
    metrics = compute_metrics(pred_mask, gt_mask)
    
    return metrics

def eval_t_iou(iou_results):
    iou_acc_results = {}
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        thr_acc = []
        for key in DISEASE_NAMES:
            iou_scores = np.array(iou_results.get(key, []), dtype=np.float32)
            if iou_scores.size == 0:
                # A finding with no boxes has no accuracy to report. Scoring it
                # 0.0 used to drag every mean down by 1/len(DISEASE_NAMES) per
                # empty class and was indistinguishable from a genuine total
                # miss. NaN says "no data", and every consumer below aggregates
                # with nanmean/nanstd so empty findings are skipped rather than
                # counted as failures.
                acc = np.nan
            else:
                iou_scores = np.where(iou_scores >= thr, 1, 0)
                acc = iou_scores.sum() / iou_scores.size
            thr_acc.append(acc)
        iou_acc_results[f'thr_{thr}'] = np.array(thr_acc, dtype=np.float32)
    return iou_acc_results
    
def eval_micro_iou_acc(iou_results):
    """Box-weighted ('micro') IoU accuracy per threshold.

    eval_t_iou averages over findings, giving each equal weight regardless of
    how many boxes it has (macro). This pools the boxes instead: every box
    counts once, so a finding contributes in proportion to its size. With this
    dataset that shifts Atelectasis from 1/7 of the weight to 303/984, and
    Nodule from 1/7 to 79/984.

    Computed straight from the per-box IoU values rather than from per-finding
    accuracies - identical arithmetic, and it sidesteps the empty-finding
    question entirely, since a finding with no boxes simply contributes none.
    """
    all_iou = [v for cls in DISEASE_NAMES for v in iou_results.get(cls, [])]
    out = {}
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        if not all_iou:
            out[f'thr_{thr}'] = float('nan')
        else:
            a = np.asarray(all_iou, dtype=np.float32)
            out[f'thr_{thr}'] = float(np.mean(a >= thr))
    out['n_boxes'] = len(all_iou)
    return out


def print_results(results):
    iou_results, dice_results, pre_results, re_results, f1_results = results

    print("\nPer-class Metrics (mean ± std):")
    for cls in findings_with_data(iou_results):
        def ms(x):
            return np.mean(x), np.std(x)

        iou_m, iou_s = ms(iou_results[cls]) if iou_results[cls] else (0, 0)
        dice_m, dice_s = ms(dice_results[cls]) if dice_results[cls] else (0, 0)
        f1_m, f1_s = ms(f1_results[cls]) if f1_results[cls] else (0, 0)
        pre_m, pre_s = ms(pre_results[cls]) if pre_results[cls] else (0, 0)
        rec_m, rec_s = ms(re_results[cls]) if re_results[cls] else (0, 0)

        print(
            f"{cls:15s} | "
            f"IoU: {iou_m:.3f}±{iou_s:.3f} | "
            f"Dice: {dice_m:.3f}±{dice_s:.3f} | "
            f"F1: {f1_m:.3f}±{f1_s:.3f} | "
            f"P: {pre_m:.3f}±{pre_s:.3f} | "
            f"R: {rec_m:.3f}±{rec_s:.3f}"
        )

    # -------- Overall --------
    def flatten(d):
        return np.array([v for lst in d.values() for v in lst])

    print("\nOverall (mean ± std):")

    for name, d in zip(
        ["IoU", "Dice", "F1-score", "Precision", "Recall"],
        [iou_results, dice_results, f1_results, pre_results, re_results]
    ):
        vals = flatten(d)
        print(f"{name:10s}: {vals.mean():.3f}±{vals.std():.3f}")

def findings_with_data(results_dict):
    """DISEASE_NAMES that actually have samples, in their canonical order.

    bbox_data.npy carries no boxes for class index 3 (Infiltrate/Infiltration)
    - the 984 boxes are distributed over the other seven findings - so that
    class is always empty and every statistic for it is vacuous. Reporting it
    as 0.000 made the model look like it failed completely on a finding it was
    never evaluated on. Excluded from the per-class tables entirely.
    """
    return [c for c in DISEASE_NAMES if len(results_dict.get(c, [])) > 0]


def iou_acc_mean_std(vals):
    """mean+-std over findings that have data, ignoring NaN (empty findings).

    Returns (nan, nan, 0) when no finding contributed, so callers can render
    that case explicitly instead of emitting a misleading 0.000.
    """
    vals = np.asarray(vals, dtype=np.float64)
    n = int(np.count_nonzero(~np.isnan(vals)))
    if n == 0:
        return float('nan'), float('nan'), 0
    return float(np.nanmean(vals)), float(np.nanstd(vals)), n


def fmt_iou_acc(vals):
    m, sd, n = iou_acc_mean_std(vals)
    if n == 0:
        return 'n/a'
    return f"{m:.3f}±{sd:.3f}"


def print_iou(iou_acc_results):
    n_all = len(DISEASE_NAMES)
    print("\nIoU Accuracy (mean ± std across findings with data):")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        vals = iou_acc_results[f'thr_{thr}']
        _, _, n = iou_acc_mean_std(vals)
        note = '' if n == n_all else f"   [{n}/{n_all} findings had boxes]"
        print(f"thr {thr}: {fmt_iou_acc(vals)}{note}")


def build_metrics_df(results):
    iou_results, dice_results, pre_results, re_results, f1_results = results

    def ms_str(x):
        if len(x) == 0:
            return f"{0.0:.3f}±{0.0:.3f}"
        return f"{np.mean(x):.3f}±{np.std(x):.3f}"

    def flatten(d):
        return np.array([v for lst in d.values() for v in lst])

    rows = []
    for cls in findings_with_data(iou_results):
        rows.append({
            'class': cls,
            'IoU': ms_str(iou_results[cls]),
            'Dice': ms_str(dice_results[cls]),
            'F1-score': ms_str(f1_results[cls]),
            'Precision': ms_str(pre_results[cls]),
            'Recall': ms_str(re_results[cls]),
        })

    rows.append({
        'class': 'Overall',
        'IoU': ms_str(flatten(iou_results)),
        'Dice': ms_str(flatten(dice_results)),
        'F1-score': ms_str(flatten(f1_results)),
        'Precision': ms_str(flatten(pre_results)),
        'Recall': ms_str(flatten(re_results)),
    })

    return pd.DataFrame(rows)


def build_iou_acc_df(iou_acc_results, iou_results=None, counts_override=None, micro=None):
    """IoU accuracy per threshold, broken out per finding and overall.

    One row per IoU threshold. Columns are one per finding - in DISEASE_NAMES
    order, which is the order eval_t_iou fills its arrays in - followed by
    'accuracy', the mean+-std across findings this table has always reported.
    The per-finding numbers were already being computed by eval_t_iou and
    then collapsed away here; this just stops discarding them.

    Pass iou_results (the raw {class: [iou, ...]} dict) to annotate each
    column header with that finding's sample count. That matters because
    eval_t_iou scores a finding with no boxes as 0.0, which on a sheet is
    indistinguishable from a genuine total miss - so findings with no samples
    are written as blank rather than 0.0 when the counts are available.
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    if iou_results is not None:
        # counts_override carries the per-fold box count. Without it we would
        # label columns with len(global_iou[cls]), which is kfolds x the real n
        # because every fold re-evaluates the same held-out test set.
        counts = (dict(counts_override) if counts_override is not None
                  else {cls: len(iou_results.get(cls, [])) for cls in DISEASE_NAMES})
        keep = findings_with_data(iou_results)
    else:
        # No counts available: fall back to whichever entries are not NaN.
        counts = None
        any_vals = iou_acc_results[f'thr_{thresholds[0]}']
        keep = [c for i, c in enumerate(DISEASE_NAMES) if not np.isnan(float(any_vals[i]))]

    rows = []
    for thr in thresholds:
        vals = iou_acc_results[f'thr_{thr}']
        row = {'threshold': thr}
        for cls in keep:
            i = DISEASE_NAMES.index(cls)
            col = cls if counts is None else f'{cls} (n={counts[cls]})'
            v = float(vals[i])
            row[col] = np.nan if np.isnan(v) else round(v, 4)
        # Deliberately last: existing readers look this column up by name.
        # It is the mean+-std across findings that had boxes - findings with
        # no samples are NaN from eval_t_iou and are skipped, not counted as
        # zeros. 'n/a' if no finding had any boxes at all.
        row['accuracy'] = fmt_iou_acc(vals)
        if micro is not None:
            row['accuracy_micro'] = round(float(micro[f'thr_{thr}']), 4)
        row['n_findings'] = int(iou_acc_mean_std(vals)[2])
        row['n_boxes'] = (int(micro['n_boxes']) if micro is not None
                          else int(sum(counts.values())) if counts else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_cross_fold_iou_acc_df(per_fold_iou_acc, per_fold_counts, per_fold_micro=None):
    """IoU accuracy per threshold as mean +- s.d. across cross-validation models.

    This is the table to report. Each fold is a separately trained model
    evaluated on the SAME held-out bounding boxes, so the spread across folds
    is model variability with the test set held constant - which is what a
    reader wants when asking how much the result depends on the trained model.

    It is NOT the pooled per-sample spread: pooling every fold's per-image IoU
    gives kfolds x n correlated values (the same images scored by each model),
    so its s.d. describes image-to-image difficulty, not uncertainty in the
    estimate, and its n overstates the sample size by a factor of kfolds.

    Per-finding cells aggregate that finding's per-fold accuracy. 'accuracy'
    aggregates each fold's own across-finding mean, so the reported s.d. is
    over folds in both cases rather than mixing the two axes.
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    counts = dict(per_fold_counts or {})
    keep = [c for c in DISEASE_NAMES if counts.get(c, 0) > 0]

    def ms(vals):
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return 'n/a'
        return f"{vals.mean():.3f}±{vals.std():.3f}"

    rows = []
    for thr in thresholds:
        key = f'thr_{thr}'
        row = {'threshold': thr}
        for cls in keep:
            i = DISEASE_NAMES.index(cls)
            row[f'{cls} (n={counts[cls]})'] = ms([f[key][i] for f in per_fold_iou_acc])
        # Each fold's own overall first, then aggregate across folds, so the
        # +- is over folds for macro and micro alike.
        row['accuracy'] = ms([iou_acc_mean_std(f[key])[0] for f in per_fold_iou_acc])
        if per_fold_micro is not None:
            row['accuracy_micro'] = ms([f[key] for f in per_fold_micro])
        # How many findings the macro average is over, and how many boxes the
        # micro average is over - so a thin average can never pass unnoticed.
        row['n_findings'] = int(np.median([iou_acc_mean_std(f[key])[2]
                                           for f in per_fold_iou_acc]))
        row['n_boxes'] = (int(per_fold_micro[0]['n_boxes'])
                          if per_fold_micro else int(sum(counts.values())))
        row['n_folds'] = len(per_fold_iou_acc)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_fold(results):
    iou_results, dice_results, pre_results, re_results, f1_results = results

    def flatten(d):
        return np.array([v for lst in d.values() for v in lst])

    return {
        "IoU": flatten(iou_results).mean(),
        "Dice": flatten(dice_results).mean(),
        "F1": flatten(f1_results).mean(),
        "Precision": flatten(pre_results).mean(),
        "Recall": flatten(re_results).mean(),
    }

if __name__ == '__main__':
    main()
