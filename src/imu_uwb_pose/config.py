import torch
from pathlib import Path
import numpy as np
import datetime
import os.path as osp

class config:
    def __init__(self, experiment=None, dataset=None, data_loc='../..'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.root_dir = Path().absolute()
        self.dataset = dataset
        self.experiment = experiment
        if self.experiment != None:
            self.checkpoint_path = self.root_dir / f"pose_models/checkpoints/{self.experiment}"
            self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        
        self.raw_amass = osp.join(data_loc, 'data/raw/amass')
        self.processed_pose = osp.join(data_loc, 'data/processed')

    torch_seed = 42

    amass_datasets = ['ACCAD', 'BMLhandball', 'BMLmovi', 'CMU',
                  'DanceDB', 'DFaust', 'EKUT', 'EyesJapanDataset', 'GRAB', 'HDM05', 'HUMAN4D', 'HumanEva', 'KIT', 'MoSh', 'PosePrior', 'SFU', 'SOMA', 'SSM', 'TCDHands', 'TotalCapture', 'Transitions']
    

    body_model = './body_models'
    absolute_joint_angles = [7, 8] # left and right joint angles
    uwb_dists = [(7,8)]
    uwb_floor_dists = [7,8]
    train_pct = 0.9

    # done with 30 fps in mind. If fps is different, change this value
    max_sample_length = 150
    batch_size = 128

    def get_smpl_skeleton(self):
        return torch.tensor([
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 8],
            [6, 9],
            [7, 10],
            [8, 11],
            [9, 12],
            [9, 13],
            [9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
        ])

