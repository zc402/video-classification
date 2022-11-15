from pathlib import Path
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from new_feature_test import VideoIO

from config.defaults import get_override_cfg


cfg = get_override_cfg()

densepose = Path(cfg.DENSEPOSE).absolute()
sys.path.append(str(densepose))

dumpfile = '/media/zc/ST16/ChaLearnIsoAll/4_IUV_New/valid/029/M_05604.pkl'
with open(dumpfile, 'rb') as f:
    results = pickle.load(f)

video = VideoIO.read_video_rgb('/media/zc/C2000Pro-1TB/ChaLearnIsoAllClass/1_Sample/valid/029/M_05604.avi')

plt.ion()
fig, ax = plt.subplots(1)
ax.xaxis.tick_top()
ax.yaxis.tick_left() 

for frame, result in zip(video, results):

    # print(result)
    box = result['pred_boxes_XYXY']
    # labels = result['pred_densepose'][0].labels

    x1, y1, x2, y2 = box[0].cpu().numpy()
    x1, x2 = x1 - 160, x2 - 160
    y1, y2 = y1 - 120, y2 - 120

    # plt.imshow((embedding[0, 0:3].cpu().permute((1,2,0)) + 5)/10)

    ax.imshow(frame)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                         edgecolor='r', facecolor="none")
 
    ax.add_patch(rect)
    plt.draw()
    plt.pause(0.1)
    plt.gca().cla()
    pass


    # CSE
    # embedding = result['pred_densepose'][0].embedding  # shape (2(person), 16, 112, 112)
    # coarse_segm = result['pred_densepose'].coarse_segm  # shape (2(person), 2, 112, 112)