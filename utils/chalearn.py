from pathlib import Path
from config.defaults import get_override_cfg

cfg = get_override_cfg()
sample_root = Path(cfg.CHALEARN.ROOT, cfg.CHALEARN.SAMPLE)

def get_labels(name_of_set):
    txt = Path(sample_root, name_of_set + '.txt')
    with txt.open('r') as f:
        label_list = f.readlines()
    labels = [line.split(' ') for line in label_list]  # [M, K, L]
    labels = [(m, k, int(l)) for (m, k, l) in labels]  # Remove \n
    return labels

train_list = get_labels('train')
test_list = get_labels('test')
val_list = get_labels('valid')

pass