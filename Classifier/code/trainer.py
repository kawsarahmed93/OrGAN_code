import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from timm.utils.model_ema import ModelEmaV2

from configs import DEVICE
from trainer_callbacks import MetricStoreBox, ExtraMetricMeter, ProgressBar

from utils import FocalLoss

import wandb
import numpy as np

def save_with_retry(save_func, path: str, retries: int=5, delay: float=2.0):
    """Runs save_func(path), retrying on transient network-filesystem errors.

    The shared storage this repo runs on intermittently reports a freshly
    created directory as missing to a subsequent low-level file open (seen
    as 'Remote I/O error' / 'Parent directory ... does not exist' even right
    after os.makedirs succeeded). Retrying after a short delay, re-creating
    the directory each time, works around that without masking a real
    missing-path bug (which would still fail after all retries).
    """
    parent_dir = os.path.dirname(path)
    last_err = None
    for attempt in range(retries):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            save_func(path)
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                print(f'\033[33;1m Save to {path} failed ({e}); retrying ({attempt + 1}/{retries})... \033[0m')
                time.sleep(delay)
    raise last_err


def check_if_best_value(current_value: float, previous_best_value: float, metric_name: str='loss', mode: str='min', verbose: bool=True):
    if mode == 'min':
        if previous_best_value > current_value:
            if verbose:
                print('\033[32;1m' + ' Val {} is improved from {:.4f} to {:.4f}! '.format(metric_name, previous_best_value, current_value) + '\033[0m')
            best_value = current_value
            is_best_value = True
        else:
            if verbose:
                print('\033[31;1m' + ' Val {} is not improved from {:.4f}! '.format(metric_name, previous_best_value) + '\033[0m')
            best_value = previous_best_value
            is_best_value = False
    else:
        if previous_best_value < current_value:
            if verbose:
                print('\033[32;1m' + ' Val {} is improved from {:.4f} to {:.4f}! '.format(metric_name, previous_best_value, current_value) + '\033[0m')
            best_value = current_value
            is_best_value = True
        else:
            if verbose:
                print('\033[31;1m' + ' Val {} is not improved from {:.4f}! '.format(metric_name, previous_best_value) + '\033[0m')
            best_value = previous_best_value
            is_best_value = False
            
    return best_value, is_best_value
      
#%% #################################### Model Trainer Class #################################### 
class ModelTrainer():
    def __init__(self, 
                 model: torch.nn.Module, 
                 Loaders: list, 
                 metrics: dict, 
                 lr: float, 
                 epochsTorun: int,
                 checkpoint_saving_path: str,
                 gpu_ids: list,
                 fold: int,
                 use_ema: bool,
                 perform_interval_validation: bool,
                 interval_validation_step: int,
                 use_wandb_log: bool=False,
                 ## problem specific parameters ##
                 pos_weight: torch.Tensor | None = None,
                 use_focal_loss: bool=False,
                 focal_loss_alpha: float=0.25,
                 focal_loss_gamma: float=2,
                 num_classes: int=14,
                 method: str='base',
                 ):
        super().__init__()
                   
        self.metrics = metrics
        self.model = model.to(DEVICE)
        self.trainLoader = Loaders[0]
        self.valLoader = Loaders[1]        
        
        self.fold = fold
        if self.fold != None:
            self.checkpoint_saving_path = checkpoint_saving_path + 'fold' + str(self.fold) + '/'
        else:
            self.checkpoint_saving_path = checkpoint_saving_path + '/'    
        os.makedirs(self.checkpoint_saving_path,exist_ok=True)
        
        self.lr = lr
        self.epochsTorun = epochsTorun       
        
        self.best_loss = 9999
        self.best_auc = -9999
        
        
        self.gpu_ids = gpu_ids
        if len(self.gpu_ids) > 1:
            print('using multi-gpu!')
            self.use_data_parallel = True
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        else:
            self.use_data_parallel = False
        
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = ModelEmaV2(self.model, decay=0.997, device=DEVICE)
        
        self.use_wandb_log = use_wandb_log
        self.perform_interval_validation = perform_interval_validation
        self.interval_validation_step = interval_validation_step
        
       
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        self.log_sigma_main = nn.Parameter(torch.zeros(1)).to(DEVICE)
        self.log_sigma_aux = nn.Parameter(torch.zeros(1)).to(DEVICE)
        
        self.all_logs = {}
        
        self.num_classes = num_classes
        if use_focal_loss:
            self.criterion_cls = FocalLoss(gamma=focal_loss_gamma, alpha=focal_loss_alpha)
        else:
            if pos_weight is not None:
                self.criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
            else:
                self.criterion_cls = nn.BCEWithLogitsLoss()
        
        self.method = method
        
    def get_checkpoint(self, val_logs):
        if self.use_ema and self.use_data_parallel:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model_ema.module.module.state_dict(),
                'Optimizer_state_dict': self.optimizer.state_dict(),
                } 
        elif self.use_ema:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model_ema.module.state_dict(),
                'Optimizer_state_dict': self.optimizer.state_dict(),
                }
        elif self.use_data_parallel:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model.module.state_dict(),
                'Optimizer_state_dict': self.optimizer.state_dict(),
                }
        else:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model.state_dict(),
                'Optimizer_state_dict': self.optimizer.state_dict(),
                }
                            
        for key in val_logs.keys():
            checkpoint_dict.update({key: val_logs[key]})
            
        return checkpoint_dict
    
    
    def perform_validation(self, use_progbar: bool=True, best_metric_verbose: bool=True):
        self.model.eval()
        if self.use_ema:
            self.model_ema.eval()
        torch.set_grad_enabled(False)
    
        val_info_box = MetricStoreBox(self.metrics)
        extra_metric_box = ExtraMetricMeter()
    
        if use_progbar:
            if self.fold is None:
                progbar_description = f'(val) Epoch {self.current_epoch_no}/{self.epochsTorun}'
            else:
                progbar_description = f'(val) Fold {self.fold} Epoch {self.current_epoch_no}/{self.epochsTorun}'
            val_progbar = ProgressBar(len(self.valLoader), progbar_description)
    
        for itera_no, data in enumerate(self.valLoader):
            images = data['image'].to(DEVICE)
            lungs = data['lung'].to(DEVICE)
            targets = data['target'].to(DEVICE).float()
    
            # ✅ FIX: correct context manager usage
            with torch.no_grad():
                if self.use_ema:
                    if self.method == 'base':
                        out = self.model_ema.module(images)
                    elif self.method == 'proposed':
                        out = self.model_ema.module(images,lungs)
                else:
                    if self.method == 'base':
                        out = self.model(images)
                    elif self.method == 'proposed':
                        out = self.model(images, lungs)
    
                batch_loss = self.criterion_cls(out['logits'], targets)
    
            # update extra metric
            y_pred = out['logits'].detach().cpu().clone().float().sigmoid().numpy()
            y_true = targets.detach().cpu().data.numpy()
            extra_metric_box.update(y_pred, y_true)
    
            # update progress bar, info box
            val_info_box.update({'loss': [batch_loss.detach().item(), targets.shape[0]]})
            logs_to_display = val_info_box.get_value()
            logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            if use_progbar:
                val_progbar.update(1, logs_to_display)
    
        # calculate all metrics
        logs_to_display = val_info_box.get_value()
        auc = extra_metric_box.feedback()
        logs_to_display.update({'auc': auc})
        logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
    
        val_logs = logs_to_display  # contains current val_loss and val_auc
    
        # ✅ BEST tracking
        self.best_loss, is_best_loss = check_if_best_value(
            val_logs['val_loss'], self.best_loss, 'loss', 'min', best_metric_verbose
        )
        self.best_auc, is_best_auc = check_if_best_value(
            val_logs['val_auc'], self.best_auc, 'auc', 'max', best_metric_verbose
        )
    
        checkpoint_dict = self.get_checkpoint(val_logs)
        if is_best_auc:
            if self.fold is None:
                save_with_retry(lambda p: torch.save(checkpoint_dict, p), self.checkpoint_saving_path + 'checkpoint_best_auc.pth')
            else:
                save_with_retry(lambda p: torch.save(checkpoint_dict, p), self.checkpoint_saving_path + f'checkpoint_best_auc_fold{self.fold}.pth')

        if is_best_loss:
            if self.fold is None:
                save_with_retry(lambda p: torch.save(checkpoint_dict, p), self.checkpoint_saving_path + 'checkpoint_best_loss.pth')
            else:
                save_with_retry(lambda p: torch.save(checkpoint_dict, p), self.checkpoint_saving_path + f'checkpoint_best_loss_fold{self.fold}.pth')
                
        del checkpoint_dict
    
        best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss': self.best_loss}
        logs_to_display.update(best_results_logs)
    
        if use_progbar:
            val_progbar.update(logs_to_display=logs_to_display)
            val_progbar.close()
    
        val_logs = logs_to_display
        if self.use_wandb_log:
            wandb.log(val_logs)
    
        return val_logs

        
    def train_one_epoch(self):
        train_info_box = MetricStoreBox(self.metrics)
        extra_metric_box = ExtraMetricMeter()
        
        if self.fold == None:
            progbar_description = f'(Train) Epoch {self.current_epoch_no}/{self.epochsTorun}'
        else:
            progbar_description = f'(Train) Fold {self.fold} Epoch {self.current_epoch_no}/{self.epochsTorun}'
        train_progbar = ProgressBar(len(self.trainLoader), progbar_description)
        
        self.model.train()
        torch.set_grad_enabled(True) 
        self.optimizer.zero_grad()
        
        if self.use_ema:
            self.model_ema.train()
        
        for itera_no, data in enumerate(self.trainLoader):                                              
            images = data['image'].to(DEVICE) 
            lungs = data['lung'].to(DEVICE) 
            targets = data['target'].to(DEVICE).float()
            
            if self.method == 'base':
                out = self.model(images)
            elif self.method == 'proposed':
                out = self.model(images, lungs)

            batch_loss = self.criterion_cls(out['logits'], targets)
            
            batch_loss.backward()
            self.optimizer.step()
            
            self.optimizer.zero_grad()
            if self.use_ema:
                self.model_ema.update(self.model)
            
            # update extra metric
            y_pred = out['logits'].detach().cpu().clone().float().sigmoid().numpy()
            y_true = targets.detach().cpu().data.numpy()
            extra_metric_box.update(y_pred, y_true)
            
            # update progress bar, info box
            train_info_box.update({'loss':[batch_loss.detach().item(), targets.shape[0]],
                                   })
            logs_to_display=train_info_box.get_value()
            auc = extra_metric_box.feedback()
            logs_to_display.update({'auc': auc})
            logs_to_display = {f'train_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
            logs_to_display.update(best_results_logs)
            train_progbar.update(1, logs_to_display)
            

            if self.perform_interval_validation:                 
                if (itera_no+1)%int(self.interval_validation_step) == 0:
                    self.perform_validation(use_progbar=False, best_metric_verbose=True)
                    self.model.train()
                    torch.set_grad_enabled(True)
            
        # calculate all metrics and close progbar
        logs_to_display=train_info_box.get_value()
        auc = extra_metric_box.feedback()
        logs_to_display.update({'auc': auc})
        logs_to_display = {f'train_{key}': logs_to_display[key] for key in logs_to_display.keys()}
        train_logs = logs_to_display
        best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
        logs_to_display.update(best_results_logs)
        train_progbar.update(logs_to_display=logs_to_display)
        train_progbar.close()
        
        if self.use_wandb_log:
            wandb.log(train_logs)
        return train_logs
            
#%% train part starts here
    def fit(self):   
        for epoch in range(self.epochsTorun):
            
            self.current_epoch_no = epoch+1
            train_logs = self.train_one_epoch()
            val_logs = self.perform_validation()
            
            self.all_logs.update({
                f'Epoch_{self.current_epoch_no}_train_logs': train_logs,
                f'Epoch_{self.current_epoch_no}_val_logs': val_logs,
                })
            save_with_retry(lambda p: np.save(p, self.all_logs), self.checkpoint_saving_path + 'all_logs.npy')