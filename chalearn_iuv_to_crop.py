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
            raise Exception(f'An image has RGB but no flow. img: {body_img_path}, expected flow: {flow_path}')
        flow_compact.append(flow)
    flow_compact = np.stack(flow_compact)  # NHWC
    # flow_compact = np.mean(flow_compact, axis=0)
    # TODO: find max of abs to replace mean operator

    # flow_uv = flow_compact[:, :, :, 0:2]  # UV. 0~255, 127 as center
    # flow_mag = flow_compact[:, :, :, 2]  # manitude. 0~255
    # nf, hf, wf, cf = flow_compact.shape
    # mag_argmax = np.argmax(flow_mag, axis=0)  # (H,W)
    # flat_flow = flow_compact.reshape(-1, cf)
    # flat_mag_argmax = mag_argmax.flatten()
    # selected_flow = np.compress(flat_mag_argmax, flat_flow, axis=0)
    # selected_flow = selected_flow.reshape(hf, wf, cf)

    return flow_compact

def crop_body(img_path:Path, target_path:Path, bbox):
    
    # if Path(target_path).exists():
    #     return  # Do not overwrite

    x1, y1, x2, y2 = bbox
    assert img_path.exists()
    img = cv2.imread(str(img_path))  # H W C
    cropped = img[y1:y2, x1:x2, :]
    cv2.imwrite(str(target_path), cropped)

    # Flow modality
    # flow_resize = cv2.resize(flow, (320, 240))  # resize to image size
    flow = load_flow(img_path)
    for i in range(flow.shape[0]):
        h, w, c = flow[i].shape
        flow_pad = np.zeros(shape=(h*2, w*2, c), dtype=img.dtype)  # pad flow
        flow_pad[h//2: h//2 + h, w//2: w//2 + w, :] = flow[i]

        crop_flow = flow_pad[y1:y2, x1:x2]
        flow_target_name = f'F{i}_' + target_path.name
        flow_target_path = Path(target_path.parent, flow_target_name)
        cv2.imwrite(str(flow_target_path), crop_flow)

    # Depth modality
    depth_folder = img_path.parent.name.replace('M_', 'K_')
    depth_path = Path(img_path.parent.parent, depth_folder, img_path.name)
    assert depth_path.exists()
    depth_img = cv2.imread(str(depth_path))
    crop_depth = depth_img[y1:y2, x1:x2, :]
    depth_target_name = 'D_' + target_path.name
    depth_target_path = Path(target_path.parent, depth_target_name)
    cv2.imwrite(str(depth_target_path), crop_depth)
    pass

    

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
        for i in range(cfg.CHALEARN.IMG_SAMPLE_INTERVAL):

            flow_img_path = Path(body_img_path.parent, f'F{i}_' + body_img_path.name)
            flow = cv2.imread(str(flow_img_path))
            flow_cropped = flow[y:y+h, x:x+w, :]
            flow_target_path = target_path.parent / (f'F{i}_' + target_path.name)
            cv2.imwrite(str(flow_target_path), flow_cropped)

        # ----------Depth
        depth_img_path = Path(body_img_path.parent, 'D_' + body_img_path.name)
        depth = cv2.imread(str(depth_img_path))
        depth_cropped = depth[y:y+h, x:x+w, :]
        depth_target_path = target_path.parent / ('D_' + target_path.name)
        cv2.imwrite(str(depth_target_path), depth_cropped)
        # ----------CSE


    [_crop_part(*args) for args in crop_part_args]  # args: (torso + larm, 'CropTorsoLArm')

    
def crop_body_bodyparts(iuv, name_of_set, pad_root, crop_body_root):
    iuv_res = load_iuv(iuv)
    
    for iuv_item in iuv_res:  # images
        # Path: ./train/001/M_00068/00000.jpg
        file_path = iuv_item['file_name']
        file_path = Path(file_path)
        x_img = file_path.name  # 00000.jpg
        x5 = file_path.parent.name  # M_00068
        if 'K_' in x5:
            print(f'warning: iuv should not parse K_ for {file_path}')
            continue
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
            box_score_argmax = np.argmax(iuv_item['scores'])
            bbox = iuv_item['pred_boxes_XYXY'][box_score_argmax].cpu().numpy().astype(int)  # shape: 4
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

    if cfg.DEBUG == True:
        for param in tqdm(param_list):
            task_wrapper(param)
    else:
        pool = Pool(min(10, cfg.NUM_CPU))  # The data is loaded into GPU, therefore 10 is largest for a 24GB memory GPU
        pool.map(task_wrapper, param_list)
        

    print(f"{name_of_set} set done")

extract_crop('train')
extract_crop('test')
extract_crop('valid')