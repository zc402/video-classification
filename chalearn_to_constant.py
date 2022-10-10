import os
import glob
from pathlib import Path
from tqdm import tqdm

ROOT_PATH = '/home/zc/Datasets/ChaLearnIsoConst'

def video_to_const_frame_rate(
    input_video:Path, 
    output_video:Path, 
    frame_rate:int):

    assert input_video.is_file()
    # Suggested by SlowFast(facebookresearch): ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
    command = f'ffmpeg -y -r {frame_rate} -i {input_video.absolute()} {output_video.absolute()}'
    execute_result = os.system(command)
    if execute_result != 0:
        print(execute_result)


def chalearn_to_const_frame_rate():
    avi_variable = glob.glob(str(Path(ROOT_PATH, '**', '*.avi')), recursive=True)  # variable frame rate
    for video in tqdm(avi_variable):
        video = Path(video)
        name = video.name  # a.avi
        new_name = 'const_' + name
        new_path = Path(video.parent, new_name)
        video_to_const_frame_rate(Path(video), Path(new_path), 10)

    for video in avi_variable:
        os.remove(video)

    avi_constant = glob.glob(str(Path(ROOT_PATH, '**', 'const_*')), recursive=True)
    for video in avi_constant:
        old_path = video
        video = Path(video)
        name = video.name
        new_name = name.replace("const_", "")
        new_path = Path(video.parent, new_name)
        os.rename(old_path, str(new_path))
    

if __name__ == '__main__':
    chalearn_to_const_frame_rate()