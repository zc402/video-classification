from pathlib import Path
from yacs.config import CfgNode

_C = CfgNode()

_C.CHALEARN = CfgNode()

_C.DEBUG = True  # Use single thread, do not save checkpoint
_C.CHALEARN.ROOT = '/media/zc/C2000Pro-1TB/ChaLearnIsoLess'  # The root of all other folders

_C.CHALEARN.NUM_CLASS = 249  # Default number of classes, labels: 1~249
_C.CHALEARN.BATCH_SIZE = 15  # Res2D: 60 Res3D: 30
_C.CHALEARN.ISO = '0_Iso'  # The folder of chalearn isogd
_C.CHALEARN.SAMPLE = '1_Sample'  # Sample gestures from the whole dataset for debugging
_C.CHALEARN.SAMPLE_CLASS = 20  # range of samples, 1~249
_C.CHALEARN.IMG = '2_Images'  # Path of converted images
_C.CHALEARN.IMG_SAMPLE_INTERVAL = 5  # Sample 1 image per 5 images
_C.CHALEARN.PAD = '3_Pad'  # Path of padded videos
_C.CHALEARN.IUV = '4_IUV'  # IUV from densepose
_C.CHALEARN.CSE = '4_CSE'  
_C.CHALEARN.CROP_BODY = 'CropBody'  # Crop body part
_C.CHALEARN.CLIP_LEN = 8  # Clip duration, Res3d
_C.CHALEARN.FLOW = '2_Flow'  # Optical flow, saved as RGB images
_C.CHALEARN.IMG_ENERGY = '2_Images_energy'  # Image with enough flow energy

_C.DENSEPOSE = './detectron2/projects/DensePose'

_C.MODEL = CfgNode()
_C.MODEL.LOGS = 'logs'
_C.MODEL.NAME = 'model_name'
_C.MODEL.CKPT_DIR = 'checkpoints'
_C.MODEL.R3D_INPUT = 'CropHTAH'  # Input for c3d model
_C.MODEL.LR = 1e-3
_C.MODEL.FUSE = True  
_C.MODEL.MAX_EPOCH = 100

_C.NUM_CPU = 16

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()

def get_override_cfg():
    cfg = get_cfg()
    override = Path('..', 'cfg_override.yaml')
    if(override.is_file()):
        cfg.merge_from_file(override)
    return cfg