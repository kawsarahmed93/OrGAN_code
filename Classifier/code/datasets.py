import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from configs import NIH_DATASET_ROOT_DIR, NIH_CXR_SINGLE_LABEL_NAMES
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(resize=256, crop=224):
    return A.Compose([
        A.Resize(height=resize, width=resize, p=1.0),
        A.RandomCrop(height=crop, width=crop, p=1.0),
        A.HorizontalFlip(p=0.5),
        # # A.Normalize(mean=[0.5], std=[0.5]),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        #     max_pixel_value=255.0,
        #     p=1.0
        # ),
        ToTensorV2(p=1.0),
    ], p=1.0, additional_targets={'lung': 'image'})

def get_valid_transforms(resize=256, crop=224):
    return A.Compose([
        A.Resize(height=resize, width=resize, p=1.0),
        A.CenterCrop(height=crop, width=crop, p=1.0),
        # # A.Normalize(mean=[0.5], std=[0.5]),
        # A.Normalize(mean=[0.485,0.456,0.406],
        #             std=[0.229,0.224,0.225],
        #             max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0, additional_targets={'lung': 'image'})

def get_test_transforms(resize=256, crop=224):
    return get_valid_transforms(resize, crop)

def get_bbox_test_transforms(resize=256, crop=224):
    return A.Compose([
        A.Resize(height=resize, width=resize, p=1.0),
        A.CenterCrop(height=crop, width=crop, p=1.0),
        # # A.Normalize(mean=[0.5], std=[0.5]),
        # A.Normalize(mean=[0.485,0.456,0.406],
        #             std=[0.229,0.224,0.225],
        #             max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'lung': 'image', 'bone': 'image', 'mask': 'image'}, bbox_params=A.BboxParams(format='pascal_voc', clip=True, min_visibility=0.0, min_area=0.0, label_fields=[]))

class NIH_IMG_LEVEL_DS(Dataset):
    def __init__(self, xray_fpaths, labels, flag, transform):
        self.xray_fpaths = xray_fpaths
        self.labels = labels
        self.transform = transform
        self.flag = flag
        
    def __len__(self):
        return len(self.xray_fpaths)
    
    def __getitem__(self, index):
        # read image
        image = Image.open(self.xray_fpaths[index]).convert('L')
        image = image.resize((512, 512), resample=Image.BILINEAR)
        image = np.array(image)/255.0

        if self.flag =="Fusion":

            #### Xray + Lung
            lung = Image.open(
                self.xray_fpaths[index].replace("images", "lungs")
            ).convert('L')
            lung = np.array(lung)/255.0

        elif self.flag =="Xrays":
            lung = np.zeros((512, 512), dtype=np.uint8)
            # lung = np.ones((512, 512), dtype=np.uint8)
            # lung = image
        elif self.flag =="Lungs":
            lung = Image.open(
                self.xray_fpaths[index].replace("images", "lungs")
            ).convert('L')
            lung = np.array(lung)/255.0
            image=lung   

        # transform image
        transformed = self.transform(image=image, lung=lung)
        transformed_image = transformed['image'].float()
        transformed_lung = transformed['lung'].float()
        
        # read label
        label = self.labels[index]
        label = torch.tensor(label).long()    
        
        return {
            'image': transformed_image, 
            'lung': transformed_lung, 
            'target': label,
            }

class NIH_IMG_BOX_LEVEL_DS(Dataset):
    def __init__(self, xray_fpaths, labels, bbox, flag, transform):
        self.xray_fpaths = xray_fpaths
        self.labels = labels
        self.bbox = bbox
        self.transform = transform
        self.flag = flag
        
    def __len__(self):
        return len(self.xray_fpaths)
    
    def __getitem__(self, index):
        # read image
        image = Image.open(self.xray_fpaths[index]).convert('L')
        image = image.resize((512, 512), resample=Image.BILINEAR)
        image = np.array(image)/255.0

        if self.flag =="Fusion":

            #### Xray + Lung
            lung = Image.open(
                self.xray_fpaths[index].replace("images", "lungs")
            ).convert('L')
            lung = np.array(lung)/255.0

        elif self.flag =="Xrays":
            lung = Image.open(
                self.xray_fpaths[index].replace("images", "lungs")
            ).convert('L')
            lung = np.array(lung)/255.0
            # lung = np.zeros((512, 512), dtype=np.uint8)
            # lung = np.ones((512, 512), dtype=np.uint8)
            # lung = image
            
        elif self.flag =="Lungs":
                    lung = Image.open(
                        self.xray_fpaths[index].replace("images", "lungs")
                    ).convert('L')
                    lung = np.array(lung)/255.0
                    image=lung   

        mask_path = self.xray_fpaths[index].replace("images", "segment").rsplit('.', 1)[0] + '.npy'
        mask = np.load(mask_path).astype(np.uint8)

        bone = Image.open(
            self.xray_fpaths[index].replace("images", "bone_suppressed")
        ).convert('L')
        bone = np.array(bone)/255.0
        
        #bbox preprocessing
        boxes = self.bbox[index].copy()
        boxes = (boxes*image.shape[0])/1024
        
        x2 = boxes[0] + boxes[3]
        y2 = boxes[1] + boxes[2]
        boxes[2] = x2
        boxes[3] = y2
        boxes = [boxes.tolist()]

        # transform image
        transformed = self.transform(image=image, lung=lung, bone=bone, mask=mask, bboxes=boxes)
        transformed_image = transformed['image'].float()
        transformed_lung = transformed['lung'].float()
        transformed_bone = transformed['bone'].float()
        transformed_mask = transformed['mask'].float()
        
        boxes = transformed['bboxes']
        if len(boxes) == 0:
            boxes = torch.tensor([0., 0., 0., 0.])
        else:
            boxes = torch.tensor(boxes[0], dtype=torch.float32)
        
        # read label
        label = self.labels[index]
        label = torch.tensor(label).long()
        
        return {
            'image': transformed_image, 
            'lung': transformed_lung, 
            'bone': transformed_bone, 
            'mask': transformed_mask, 
            'target': label,
            'bbox': boxes,
            }

def collate_fn_img_level_ds(batch):
    x = batch[0]
    keys = x.keys()
    out = {}
    # declare key
    for key in keys:
        out.update({key:[]})
    # append values
    for i in range(len(batch)):
        for key in keys:
            out[key].append(batch[i][key])
    # stack values
    for key in keys:
        out[key] = torch.stack(out[key])
    
    return out

if __name__ == '__main__':
    train_df = pd.read_csv('./LongTailCXR/nih-cxr-lt_single-label_train.csv')
    train_fpaths = np.array([NIH_DATASET_ROOT_DIR + x for x in train_df['id'].values])
    train_labels = np.stack([np.array(train_df[x]) for x in NIH_CXR_SINGLE_LABEL_NAMES], axis=1).argmax(1) 
    
    train_dataset = NIH_IMG_LEVEL_DS(
                        train_fpaths,
                        train_labels,
                        get_train_transforms(256, 224),
                        )
    data = train_dataset[25]
    image, label = data['image'], data['target']
    plt.imshow(image[0], cmap='gray');plt.axis('off');plt.show();