from yacs.config import CfgNode

_C = CfgNode()

_C.CHALEARN = CfgNode()

_C.CHALEARN.ROOT = '/home/zc/Datasets/ChaLearnIsoConst/'
_C.CHALEARN.NUM_CLASS = 249  # labels: 1~249
_C.CHALEARN.BATCH_SIZE = 4

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()