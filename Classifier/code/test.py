from argparse import ArgumentParser

import os

import torch
from torch.utils.data import DataLoader
from torch import nn
import time
import cv2

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, average_precision_score

import pandas as pd

from datasets import NIH_IMG_LEVEL_DS, get_valid_transforms, collate_fn_img_level_ds
from models import DenseNet121, FusionNet
from configs import all_configs, DEVICE, NIH_DATASET_ROOT_DIR, LABEL_DIR, NIH_CXR_SINGLE_LABEL_NAMES
from trainer_callbacks import set_random_state

# Prevent each DataLoader worker process from also spawning its own OpenCV
# thread pool (albumentations' transforms are cv2-backed under the hood) -
# with num_workers>0 that oversubscribes CPUs and can make preprocessing
# slower, not faster, on a shared node.
cv2.setNumThreads(0)


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

    Mirrors the helper of the same name in test_iou.py and the OrGAN train-*
    scripts.
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
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['xray_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['lung_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['bone_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['segment_base'])
    parser.add_argument('--run_configs_list', type=str, nargs="*", default=['proposed'])
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--image_resize_dim', type=int, default=586)
    # parser.add_argument('--image_crop_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=15)
    args = parser.parse_args()
    return args

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

    fold_auc_per_label = []

    for fold in range(kfolds):
    
        split_dict = np.load(f"{LABEL_DIR}{fold}.npy", allow_pickle=True).item()
        
        test_fpaths = np.array([NIH_DATASET_ROOT_DIR+x for x in split_dict['test_fnames']])
        test_labels = np.array(split_dict['test_labels'])
    
        # get dataloader
        print('Loading Baseline dataloader!')
        test_dataset = NIH_IMG_LEVEL_DS(
                            test_fpaths,
                            test_labels,
                            configs['flag'],
                            get_valid_transforms(configs['resize_crop'][0], configs['resize_crop'][1]),
                            )  
             
        test_loader = DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_workers,
                            drop_last=False,
                            collate_fn=collate_fn_img_level_ds,
                            pin_memory=True,
                            persistent_workers=args.n_workers > 0,
                            )
         
        all_targets = []
        all_preds = []
            
        print('Loading model!')
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
    
        torch.set_grad_enabled(False)
        with torch.no_grad():
            for itera_no, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                images = data['image'].to(DEVICE) 
                lungs = data['lung'].to(DEVICE) 
                targets = data['target'].to(DEVICE).float()
                
                if configs['model_type'] == 'densenet121':
                    out = model(images)
                elif configs['model_type'] == 'fusion_net':
                    out = model(images,lungs)
                    
                all_targets.append(targets.cpu().data.numpy())              
                y_pred = out['logits'].cpu().detach().clone().float().sigmoid()
                all_preds.append(y_pred.numpy())
                
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        auc_per_label = roc_auc_score(all_targets, all_preds, average=None)
        fold_auc_per_label.append(auc_per_label)
        print(f'Fold: {fold} - Auc score:')
        for i, auc in enumerate(auc_per_label):
            print(f"{NIH_CXR_SINGLE_LABEL_NAMES[i]}: {auc:.4f}")
    
        mean_auc = np.nanmean(auc_per_label[:-1])
        print(f'Overall AUC: {mean_auc:.4f} \n')

        # pr_auc_per_label = average_precision_score(all_targets, all_preds, average=None)

        # print(f'Fold: {fold} - PR AUC score:')
        # for i, pr_auc in enumerate(pr_auc_per_label):
        #     print(f"{NIH_CXR_SINGLE_LABEL_NAMES[i]}: {pr_auc:.4f}")
        
        # mean_pr_auc = np.nanmean(pr_auc_per_label[:-1])
        # print(f'Overall PR AUC: {mean_pr_auc:.4f} \n')

    # Standard k-fold CV reporting: mean +- std computed across the k
    # fold-level AUCs (not a pooled/out-of-fold single estimate). This is
    # what actually characterizes how stable the estimate is under
    # different train/test splits.
    fold_auc_arr = np.array(fold_auc_per_label)  # (kfolds, num_labels)
    label_mean = np.nanmean(fold_auc_arr, axis=0)
    label_std = np.nanstd(fold_auc_arr, axis=0)

    fold_overall_auc = np.nanmean(fold_auc_arr[:, :-1], axis=1)  # per-fold overall AUC
    overall_mean = fold_overall_auc.mean()
    overall_std = fold_overall_auc.std()

    print("\nCross-Fold AUC per label (mean±std across folds):")
    for name, m, s in zip(NIH_CXR_SINGLE_LABEL_NAMES, label_mean, label_std):
        print(f"{name}: {m:.4f}±{s:.3f}")

    print(f"\nCross-Fold Overall AUC: {overall_mean:.4f}±{overall_std:.3f}")

    results_df = pd.DataFrame({
        'label': NIH_CXR_SINGLE_LABEL_NAMES,
        'auc': [f"{m:.4f}±{s:.3f}" for m, s in zip(label_mean, label_std)],
    })
    results_df = pd.concat([
        results_df,
        pd.DataFrame([{
            'label': 'Overall',
            'auc': f"{overall_mean:.4f}±{overall_std:.3f}",
        }]),
    ], ignore_index=True)

    excel_path = weight_saving_path + f'global_auc_{run_config}.xlsx'
    save_with_retry(lambda p: results_df.to_excel(p, index=False), excel_path)
    print(f"Saved global AUC results to {excel_path}")

        # all_preds = (all_preds >= 0.5).astype(int)
    
        # print(classification_report(
        #     all_targets,
        #     all_preds,
        #     target_names=NIH_CXR_SINGLE_LABEL_NAMES  # optional list of label names
        # ))
        
        # print("Macro-F1:", f1_score(all_targets, all_preds, average='macro'))
        # print("Weighted-F1:", f1_score(all_targets, all_preds, average='weighted'))
        # print("Micro-F1:", f1_score(all_targets, all_preds, average='micro'))
        # print("Accuracy:", accuracy_score(all_targets, all_preds))
        # print("Balanced accuracy:", balanced_accuracy_score(all_targets, all_preds))
        # print(classification_report(all_targets, all_preds, digits=4))
        
        # print(confusion_matrix(all_targets, all_preds))
    
        
if __name__ == '__main__':
    main()