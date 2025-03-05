r"""
imu_uwb_pose Model
"""

import torch.nn as nn
import torch
import pytorch_lightning as pl
from .RNN import RNN
from imu_uwb_pose import config
from imu_uwb_pose.utils import default_smpl_input
import smplx

class imu_uwb_pose_model(pl.LightningModule):
    r"""
    Inputs - global orientation and uwb distances, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config:config):
        super().__init__()
        n_input = 3 * len(config.absolute_joint_angles) + len(config.uwb_dists) + len(config.uwb_floor_dists)

        n_output_joints = 23  # 21 joints + 1 (root) + 1 translation
        self.n_output_joints = n_output_joints
        self.n_pose_output = n_output_joints * 3

        n_output = self.n_pose_output

        self.batch_size = config.batch_size
        
        self.model = RNN(n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)

        self.config = config
        
        self.body_model = smplx.create(config.body_model, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         batch_size=1,
                         ext='npz',
                         age='adult').to(config.device)

        self.loss = nn.MSELoss()
        self.lr = 3e-4
        self.save_hyperparameters()
        
        self.validation_step_outputs = []  # Store validation outputs manually

    def forward(self, inputs, lens):
        pred_pose, _, _ = self.model(inputs, lens)
        return pred_pose

    def step(self, batch):
        inputs, target_pose, target_joints, input_lengths, _ = batch
        
        pred = self(inputs, input_lengths)
        
        target = target_pose
        
        loss = self.loss(pred, target_pose)
        loss += self.calculate_joint_loss(pred, target_joints, input_lengths)
        
        return loss, pred, target

    def calculate_joint_loss(self, pred_pose, target_joints, lens):
        batch_size = len(target_joints)
        if batch_size == 0:
            return torch.tensor(0.0, device=self.config.device)

        pred_pose = pred_pose.reshape(-1, self.n_pose_output) 

        smpl_input = default_smpl_input(pred_pose.shape[0], self.config)
        smpl_input['global_orient'] = pred_pose[:, :3]
        smpl_input['body_pose'] = pred_pose[:, 3:66]
        smpl_input['transl'] = pred_pose[:, 66:]
        pred_joints = self.body_model(**smpl_input).joints[:, 0:22, :]

        pred_joints = pred_joints.reshape(batch_size, self.config.max_sample_length, -1, 3)
        
        loss = self.loss(pred_joints, target_joints)
        return loss

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        
        self.log("validation_step_loss", loss.item(), batch_size=self.batch_size)
        self.validation_step_outputs.append({"loss": loss.item()})  # Collect outputs manually
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        loss, pred_pose, target_pose = self.step(batch)
        
        return {"loss": loss.item(), "pred": pred_pose, "true": target_pose}
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is not None:
            self.validation_step_outputs.append(outputs)
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("training_step_loss")
        if avg_loss is not None:
            self.log("train_loss_epoch", avg_loss, prog_bar=True)
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            self.epoch_end_callback(self.validation_step_outputs, loop_type="val")
        self.validation_step_outputs.clear()  # Clear stored outputs

    def on_test_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("test_loss")
        if avg_loss is not None:
            self.log("test_loss_epoch", avg_loss, prog_bar=True)

    def epoch_end_callback(self, outputs, loop_type="train"):
        loss = [output["loss"] for output in outputs]
        avg_loss = torch.mean(torch.tensor(loss))
        self.log(f"{loop_type}_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
