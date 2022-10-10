from pathlib import Path

txt = Path('/home/zc/Datasets/ChaLearnIsoConst/IsoGD_labels/train.txt')
with txt.open('r') as f:
    lines = f.readlines()

labels = [int(l.split(' ')[2]) for l in lines]

print(min(labels), max(labels), len(set(labels)))
print(f'num of videos: {len(labels)}')