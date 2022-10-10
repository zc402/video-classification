from pathlib import Path
from yacs.config import CfgNode

_C = CfgNode()

_C.CHALEARN = CfgNode()

_C.CHALEARN.ROOT = '/home/zc/Datasets/ChaLearn/ChaLearn Isolated Gesture Recognition'
_C.CHALEARN.NUM_CLASS = 249  # labels: 1~249
_C.CHALEARN.BATCH_SIZE = 4
_C.CHALEARN.IMG_ROOT = '/home/zc/Datasets/ChaLearn/ChaLearnIsoImages'  # Path of converted images
_C.CHALEARN.PAD_ROOT = '/home/zc/Datasets/ChaLearn/ChaLearnIsoPad'  # Path of padded videos


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