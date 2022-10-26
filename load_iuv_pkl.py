import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

sys.path.append("/home/zc/NutstoreFiles/Projects/deep-learning/detectron2/projects/DensePose/")

dumpfile = '/media/zc/C2000Pro-1TB/ChaLearnIsoLess/4_CSE/train/007.pkl'
with open(dumpfile, 'rb') as f:
    result = pickle.load(f)

# print(result)
embedding = result[0]['pred_densepose'].embedding  # shape (2(person), 16, 112, 112)
coarse_segm = result[0]['pred_densepose'].coarse_segm  # shape (2(person), 2, 112, 112)

# plt.imshow((embedding[0, 0:3].cpu().permute((1,2,0)) + 5)/10)
plt.imshow(coarse_segm[0, 1:2].cpu().permute((1,2,0)))
plt.show()
pass
