WANDB_PROJECT = "BDD100k"
ENTITY = None # 'av-team'
BDD_CLASSES = {i:c for i,c in enumerate(['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle'])}
RAW_DATA_AT = 'bdd_simple_1k'
PROCESSED_DATA_AT = 'bdd_simple_1k_split'