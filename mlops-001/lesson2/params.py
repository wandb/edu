WANDB_PROJECT = "BDD100k"
ENTITY = None # 'av-team'
BDD_CLASSES = {i:c for i,c in enumerate(['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
])}
RAW_DATA_AT = 'bdd_sample_1k'
PROCESSED_DATA_AT = 'bdd_sample_1k_split'