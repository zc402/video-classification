import sys
import numpy as np
import pickle

sys.path.append("/home/zc/NutstoreFiles/Projects/deep-learning/detectron2/projects/DensePose/")

dumpfile = '/home/zc/NutstoreFiles/Projects/deep-learning/detectron2/projects/DensePose/dump.pkl'
with open(dumpfile, 'rb') as f:
    result = pickle.load(f)

print(result)
pass
