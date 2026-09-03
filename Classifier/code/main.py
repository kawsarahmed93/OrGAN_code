from argparse import ArgumentParser

import cv2
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from trainer import ModelTrainer

import pandas as pd
from datasets import NIH_IMG_LEVEL_DS, get_train_transforms, get_valid_transforms, get_test_transforms, collate_fn_img_level_ds
from models import DenseNet121, FusionNet
from configs import all_configs, NIH_DATASET_ROOT_DIR, LABEL_DIR
from trainer_callbacks import set_random_state, AverageMeter, PrintMeter
from sklearn.model_selection import train_test_split

import wandb
import torch.nn as nn

torch.backends.cudnn.benchmark = True
# Prevent each DataLoader worker process from also spawning its own OpenCV
# thread pool - albumentations (used in datasets.py) wraps cv2 internally, so
# with num_workers>0 that oversubscribes CPUs and can make augmentation
# slower, not faster, on a shared node.
cv2.setNumThreads(0)



def get_args():
    """
    get command line args
    """
    parser = ArgumentParser(description='Classification_Model')
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['xray_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['lung_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['bone_base'])
    # parser.add_argument('--run_configs_list', type=str, nargs="*", default=['segment_base'])
    parser.add_argument('--run_configs_list', type=str, nargs="*", default=['proposed'])
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=4690)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--image_resize_dim', type=int, default=586)
    # parser.add_argument('--image_crop_dim', type=int, default=512)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--perform_interval_validation', type=bool, default=True)
    parser.add_argument('--interval_validation_step', type=int, default=200)
    parser.add_argument('--use_wandb_log', type=bool, default=False)
    parser.add_argument('--use_focal_loss', type=bool, default=False)
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25)
    parser.add_argument('--focal_loss_gamma', type=float, default=2)
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--resume-fold', type=int, default=0,
                         help='Fold index to resume CV training from (skips folds before this one, e.g. 3 to resume after folds 0-2 completed)')
    args = parser.parse_args()
    return args

def main():
    """
    main function
    """

    args = get_args()
    kfolds=5

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)
            
    # check if there are duplicate weight saving paths
    unique_paths = np.unique([ x[1]['weight_saving_path'] for x in all_configs.items() ])
    assert len(all_configs.keys()) == len(unique_paths)
    
    assert 0 <= args.resume_fold < kfolds, f'--resume-fold must be in [0, {kfolds - 1}], got {args.resume_fold}'

    for config_name in args.run_configs_list:
        configs = all_configs[config_name]
        # print(configs)
        set_random_state(args.seed)

        if args.use_wandb_log:
            ## wandb part
            wandb_log_configs = vars(args)
            wandb_log_configs.update(configs)
            wandb.init(
                project="NIH_CXR_DS_RUNS",
                config=wandb_log_configs,
            )

        if args.resume_fold > 0:
            print(f'Resuming CV from fold {args.resume_fold} (skipping folds 0-{args.resume_fold - 1})')

        for fold in range(args.resume_fold, kfolds):

            split_dict = np.load(f"{LABEL_DIR}{fold}.npy", allow_pickle=True).item()
            
            set_random_state(args.seed)
            
            train_fpaths = np.array([NIH_DATASET_ROOT_DIR+x for x in split_dict['train_fnames']])
            train_labels = np.array(split_dict['train_labels'])
            val_fpaths = np.array([NIH_DATASET_ROOT_DIR+x for x in split_dict['val_fnames']])
            val_labels = np.array(split_dict['val_labels'])
    
            
            # # ---- WeightedRandomSampler (balance classes in batches) ----
            num_classes = args.num_classes
    
            class_counts = train_labels.sum(axis=0).astype(np.float32)
            class_weights = 1.0 / (class_counts + 1e-6)
    
            pos_weight = torch.tensor(
                (len(train_labels) - class_counts) / (class_counts + 1e-6)
            )
            
            train_dataset = NIH_IMG_LEVEL_DS(
                                train_fpaths,
                                train_labels,
                                configs['flag'],
                                get_train_transforms(configs['resize_crop'][0], configs['resize_crop'][1]),
                                )
            val_dataset = NIH_IMG_LEVEL_DS(
                                val_fpaths,
                                val_labels,
                                configs['flag'],
                                get_valid_transforms(configs['resize_crop'][0], configs['resize_crop'][1]),
                                )
    
            train_loader = DataLoader(
                                train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.n_workers,
                                drop_last=True,
                                collate_fn=collate_fn_img_level_ds,
                                pin_memory=True,
                                persistent_workers=args.n_workers > 0,
                                )

            val_loader = DataLoader(
                                val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_workers,
                                drop_last=False,
                                collate_fn=collate_fn_img_level_ds,
                                pin_memory=True,
                                persistent_workers=args.n_workers > 0,
                                )
            
            print('Loading model!')
            if configs['model_type'] == 'densenet121':
                model = DenseNet121(args.num_classes)
            elif configs['model_type'] == 'fusion_net':
                model = FusionNet(args.num_classes)
                
            trainer_args = {
                    'model': model,
                    'Loaders': [train_loader, val_loader],
                    'metrics': {
                        'loss': AverageMeter,
                        'auc': PrintMeter,
                        },
                    'checkpoint_saving_path': configs['weight_saving_path'],
                    'lr': args.lr,
                    'epochsTorun': configs['epochs'],
                    'gpu_ids': args.gpu_ids,
                    'fold': fold,
                    'use_ema': args.use_ema,
                    'perform_interval_validation': args.perform_interval_validation,
                    'interval_validation_step': args.interval_validation_step,
                    'use_wandb_log': args.use_wandb_log,
                    ## problem specific parameters ##
                    # 'pos_weight': pos_weight,
                    'pos_weight': None,
                    'use_focal_loss': args.use_focal_loss,
                    'focal_loss_alpha': args.focal_loss_alpha,
                    'focal_loss_gamma': args.focal_loss_gamma,
                    'num_classes': args.num_classes, # NO FUNCTION in Trainer
                    'method': configs['method'],
                    }
    
            trainer = ModelTrainer(**trainer_args)
            trainer.fit()
            
            if args.use_wandb_log:
                wandb.finish()
            
if __name__ == '__main__':
    main()  