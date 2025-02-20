import torch
from pathlib import Path

class config:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.root_dir = Path().absolute()
        
    torch_seed = 0
    amass_datasets = ['HumanEva']
    raw_amass = './data/raw/amass'
    processed_pose = './data/processed'

