import torch
from pathlib import Path
import numpy as np
import datetime
import os

class config:
    def __init__(self, experiment=None, dataset=None, lr=1e-3, name=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.root_dir = Path().absolute()
        self.dataset = dataset
        self.experiment = experiment
        self.name = name
        self.lr = lr
        self.imu_uwb_pose_model_path = Path('/media/lichard/8E98F15098F136F5/imu_uwb_pose_models')
        if self.experiment != None:
            self.checkpoint_path = self.imu_uwb_pose_model_path / f"pose_models/checkpoints/{self.experiment}"
            if self.name:
                self.checkpoint_path = self.checkpoint_path / f'{self.name}'
            self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        self.fig_path = self.root_dir / f'figs'
        self.fig_path.mkdir(exist_ok=True, parents=True)

        slurm_workers = os.environ.get("SLURM_CPUS_PER_TASK")
        fallback_workers = os.cpu_count() or 8
        self.num_workers = int(slurm_workers) if slurm_workers else fallback_workers
        self.persistent_workers = self.num_workers > 0
        self.pin_memory = self.device.type == 'cuda'

        
    torch_seed = 42
    amass_datasets = ['ACCAD', 'BMLmovi', 'CMU',
                  'DanceDB', 'DFaust', 'EKUT', 'Eyes_Japan_Dataset', 'HDM05', 'HUMAN4D', 'HumanEva', 'KIT', 'MoSh', 'PosePrior', 'SFU', 'SOMA', 'SSM', 'TCDHands', 'TotalCapture', 'Transitions']
    
    raw_amass = '/media/lichard/8E98F15098F136F5/amass'
    raw_footposer = './data/raw/FootPoser_filtered'
    processed_pose = './data/processed'
    body_model = './body_models'
    absolute_joint_angles = [7, 8] # left and right joint angles
    uwb_dists = [(7,8)]
    uwb_floor_dists = [7,8]
    acceleration_joints = [7,8]
    train_pct = 0.9

    # done with 30 fps in mind. If fps is different, change this value
    max_sample_length = 150
    batch_size = 32

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

    motion_uids = {
        'activities': 0,
        'gestures': 1,
        'exercises': 2,
    }

    participant_uids = {
        'evan': 0,
        'helen': 1,
        'jack': 2,
        'jason': 3,
        'jin': 4,
        'kanav': 5,
        'maggie': 6,
        'michelle': 7,
        'vidya': 8,
    }
