from pathlib import Path
import sys
import numpy as np
import pickle
from config.defaults import get_override_cfg
from tqdm import tqdm
import cv2
import glob 
from utils.chalearn import train_list, test_list
from config.crop_cfg import crop_part_args
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
from multiprocessing import Pool

cfg = get_override_cfg()

densepose = Path(cfg.DENSEPOSE)
sys.path.append(str(densepose))

def load_iuv(pkl_path):
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result

def load_flow(body_img_path):
    img_num = int(body_img_path.stem)
    flow_start = img_num - cfg.CHALEARN.IMG_SAMPLE_INTERVAL+1  
    flow_end = img_num
    flow_num = list(range(flow_start, flow_end + 1))  # -4,...,0
    flow_num = [max(i, 0) for i in flow_num]
    flow_name = [str(i).zfill(5) for i in flow_num]
    flow_name = [i+'.jpg' for i in flow_name]

    name_set, XXX, M_XXXXX  = body_img_path.parent.parts[-3:]  # ('train', '159', 'M_31633')
    base = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.FLOW)
    flow_folder = Path(base, name_set, XXX, M_XXXXX)
    flow_compact = []
    for name in flow_name:  # 00001.jpg,  ...,  00005.jpg
        flow_path = Path(flow_folder, name)
        # anchor_path = Path(flow_folder, flow_name[-1])  # Should always exist
        if flow_path.exists():
            flow = cv2.imread(str(flow_path))
        else:
            # TODO: fix 00000.jpg (rerun video to flow) and raise exception here
            flow = np.zeros((60, 80, 3), dtype=np.uint8) + 127
        flow_compact.append(flow)
    flow_compact = np.stack(flow_compact)  # NHWC
    flow_compact = np.mean(flow_compact, axis=0)
    # TODO: find max of abs to replace mean operator
    # flow_uv = flow_compact[:, :, :, 0:2]  # UV. 0~255, 127 as center
    # flow_mag = flow_compact[:, :, :, 2:3]  # manitude. 0~255
    # flow_uv_0 = flow_uv.astype(np.float) - 127  # -127 ~ 128, 0 as center
    # abs_uv = np.abs(flow_uv_0)
    # uv_argmax = np.argmax(abs_uv, axis=0)
    # max_flow_uv = flow_uv[uv_argmax, :, :, :]


    return flow_compact

def crop_body(img_path, target_path, bbox):
    flow = load_flow(img_path)  # (60, 80, 3)
    # if Path(target_path).exists():
    #     return  # Do not overwrite
    # if bbox is None:
    #     black = np.zeros((10, 10, 3), dtype=np.uint8)
    #     cv2.imwrite(str(target_path), black)
    #     return
    x1, y1, x2, y2 = bbox
    img = cv2.imread(str(img_path))  # H W C
    cropped = img[y1:y2, x1:x2, :]
    cv2.imwrite(str(target_path), cropped)
    # crop flow 
    
    flow_resize = cv2.resize(flow, (320, 240))  # resize to image size
    h, w, c = flow_resize.shape
    flow_pad = np.zeros(shape=(h*2, w*2, c), dtype=img.dtype) + 127  # pad flow
    flow_pad[h//2: h//2 + h, w//2: w//2 + w, :] = flow_resize
    
    crop_flow = flow_pad[y1:y2, x1:x2]
    flow_target_name = 'F_' + target_path.name
    flow_target_path = Path(target_path.parent, flow_target_name)
    cv2.imwrite(str(flow_target_path), crop_flow)
    


    

def crop_body_parts(body_img_path, target_relative_path, iuv):
    """
    Crop body parts from detected body image.

    body_img_path: path of detected and cropped human image
    """

    I = iuv['pred_densepose'][0].labels.cpu().numpy()  # pixel region segmentation results
    UV = iuv['pred_densepose'][0].uv.cpu().numpy()  # [0, 1]

    def _crop_part(part_indices, save_name):

        target_path = Path(cfg.CHALEARN.ROOT, save_name, target_relative_path)
        # if Path(target_path).exists():
        #     return  # Do not overwrite
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # mask = (I == part_indices)
        mask = np.zeros_like(I)
        for pid in part_indices:
            part_mask = (I == pid)
            mask = np.logical_or(mask, part_mask)

        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            return         
        # ---- len >= 1 ----
        img = cv2.imread(str(body_img_path))
        # Show-------------
        show = False
        if show:
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                rect = patches.Rectangle((x, y), w, h, linewidth=1,
                            edgecolor='r', facecolor="none")
                ax.add_patch(rect)
            plt.show()
            plt.close()
        # Write----------------
        area = []
        xywh = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area.append(w*h)
            xywh.append((x,y,w,h))
        amax = np.array(area).argmax()
        largest_xywh = xywh[amax]
        x,y,w,h = largest_xywh
        if w < 15 or h < 15:
            return  # Discard images with abnormal small size
        # ----------cut image from CropBody
        cropped = img[y:y+h, x:x+w, :]
        cv2.imwrite(str(target_path), cropped)
        # ----------UV
        U_crop = UV[0][y:y+h, x:x+w] * 256.   # 0~1 f -> 0~255 f
        V_crop = UV[1][y:y+h, x:x+w] * 256.

        U_crop = U_crop.astype(np.uint8)
        V_crop = V_crop.astype(np.uint8)

        U_path = target_path.parent / ('U_' + target_path.name)
        V_path = target_path.parent / ('V_' + target_path.name)
        cv2.imwrite(str(U_path), U_crop)
        cv2.imwrite(str(V_path), V_crop)

        # ----------Flow
        flow_img_path = Path(body_img_path.parent, 'F_' + body_img_path.name)
        flow = cv2.imread(str(flow_img_path))
        flow_cropped = flow[y:y+h, x:x+w, :]
        flow_target_path = target_path.parent / ('F_' + target_path.name)
        cv2.imwrite(str(flow_target_path), flow_cropped)

    [_crop_part(*args) for args in crop_part_args]  # args: (torso + larm, 'CropTorsoLArm')

    
def crop_body_bodyparts(iuv, name_of_set, pad_root, crop_body_root):
    iuv_res = load_iuv(iuv)
    
    for iuv_item in iuv_res:  # images
        # Path: ./train/001/M_00068/00000.jpg
        file_path = iuv_item['file_name']
        file_path = Path(file_path)
        x_img = file_path.name  # 00000.jpg
        x5 = file_path.parent.name  # M_00068
        x3 = Path(iuv).stem  # 001
        x3x5img = Path(x3, x5, x_img)  # 001/M_00068/00000.jpg
        nsetx3x5img = Path(name_of_set, x3x5img)
        pad_img_path = Path(pad_root, name_of_set, x3x5img)
        crop_img_path = Path(crop_body_root, name_of_set, x3x5img)
        # if crop_img_path.exists():
        #     continue  # Do not override
        crop_img_path.parent.mkdir(parents=True, exist_ok=True)

        if iuv_item['pred_boxes_XYXY'].size()[0] == 0:
            # No detection
            # crop(pad_img_path, crop_img_path, None)
            print(f"No box detection: {pad_img_path}")
        else:
            bbox = iuv_item['pred_boxes_XYXY'].cpu().numpy().astype(int)[0]  # shape: 4
            crop_body(pad_img_path, crop_img_path, bbox)
            crop_body_parts(crop_img_path, nsetx3x5img, iuv_item)

def task_wrapper(param):
    iuv, name_of_set, pad_root, crop_body_root = param
    crop_body_bodyparts(iuv, name_of_set, pad_root, crop_body_root)

def extract_crop(name_of_set):
    pad_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.PAD)
    iuv_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.IUV)
    crop_body_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.CROP_BODY)

    iuv_list = glob.glob(str(Path(iuv_root, name_of_set, "*.pkl")))

    param_list = [(iuv, name_of_set, pad_root, crop_body_root) for iuv in iuv_list]

    pool = Pool(15)
    pool.map(task_wrapper, param_list)

    # Alternative
    # for param in tqdm(param_list):
    #     task_wrapper(param)

    print(f"{name_of_set} set done")


extract_crop('train')
extract_crop('test')
extract_crop('valid')