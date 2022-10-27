from pathlib import Path
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

from config.defaults import get_override_cfg

cfg = get_override_cfg()

densepose = Path(cfg.DENSEPOSE).absolute()
sys.path.append(str(densepose))

dumpfile = '/media/zc/C2000Pro-1TB/ChaLearnIsoLess/4_IUV/train/029.pkl'
with open(dumpfile, 'rb') as f:
    result = pickle.load(f)

# print(result)
embedding = result[0]['pred_densepose'].embedding  # shape (2(person), 16, 112, 112)
coarse_segm = result[0]['pred_densepose'].coarse_segm  # shape (2(person), 2, 112, 112)

# plt.imshow((embedding[0, 0:3].cpu().permute((1,2,0)) + 5)/10)
plt.imshow(coarse_segm[0, 1:2].cpu().permute((1,2,0)))
plt.show()
pass
