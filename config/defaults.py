from pathlib import Path
from yacs.config import CfgNode

_C = CfgNode()

_C.CHALEARN = CfgNode()

_C.CHALEARN.ROOT = '/home/zc/Datasets/ChaLearnIso'  # The root of all other folders

_C.CHALEARN.NUM_CLASS = 249  # labels: 1~249
_C.CHALEARN.BATCH_SIZE = 4
_C.CHALEARN.ISO = 'Iso'  # The folder of chalearn isogd
_C.CHALEARN.SAMPLE = 'Sample'  # Sample gestures from the whole dataset for debugging
_C.CHALEARN.SAMPLE_CLASS = 5  # range of samples, 1~249
_C.CHALEARN.IMG = 'Images'  # Path of converted images
_C.CHALEARN.IMG_SAMPLE_INTERVAL = 5  # Sample 1 image per 5 images
_C.CHALEARN.PAD = 'Pad'  # Path of padded videos
_C.CHALEARN.IUV = 'IUV'  # IUV from densepose
_C.CHALEARN.CROP_BODY = 'CropBody'  # Crop body part
_C.CHALEARN.CLIP = 8  # Clip duration

_C.DENSEPOSE = '/home/zc/NutstoreFiles/Projects/deep-learning/detectron2/projects/DensePose'  # base dir of apply_net.py

_C.MODEL = CfgNode()
_C.MODEL.CKPT = './logs/checkpoints/model.ckpt'

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