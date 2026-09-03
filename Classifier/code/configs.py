import torch

DEVICE = torch.device("cuda")

NIH_DATASET_ROOT_DIR = '../NIH-CXR/images/'

NIH_CXR_SINGLE_LABEL_NAMES = ['Atelectasis', 
                             'Cardiomegaly', 
                             'Effusion', 
                             'Infiltration', 
                             'Mass', 
                             'Nodule', 
                             'Pneumonia', 
                             'Pneumothorax', 
                             'Consolidation', 
                             'Edema', 
                             'Emphysema', 
                             'Fibrosis', 
                             'Pleural_Thickening', 
                             'Hernia', 
                             'No Finding']


LABEL_DIR = './label_info/kfold_split/NIH_img_level_split_info_dict_oct_2023_split_ratio_rs1_4690_rs2_1234_fold'

all_configs = {

    # Chest X-ray only (single-stream DenseNet-121 baseline).
    'xray_base': {
        'weight_saving_path': '../weights_cv/D121_Xrays_224_bce/',
        'epochs': 20,
        'checkpoint_path': None,
        'method': 'base',
        'model_type': 'densenet121',
        'flag': 'Xrays',
        'pc': 80,
        'resize_crop': [256, 224],
    },

    # OrGAN-generated lung image only (single-stream DenseNet-121 baseline).
    'lung_base': {
        'weight_saving_path': '../weights_cv/D121_Lungs_224_bce/',
        'epochs': 20,
        'checkpoint_path': None,
        'method': 'base',
        'model_type': 'densenet121',
        'flag': 'Lungs',
        'pc': 80,
        'resize_crop': [256, 224],
    },

    # Proposed: two-stream gated fusion of the chest X-ray and the
    # OrGAN-generated lung image.
    'proposed': {
        'weight_saving_path': '../weights_cv/FusionNet_xl_224_bce/',
        'epochs': 20,
        'checkpoint_path': None,
        'method': 'proposed',
        'model_type': 'fusion_net',
        'flag': 'Fusion',
        'pc': 80,
        'resize_crop': [256, 224],
    },
}
