r"""
imu_uwb_pose Model
"""

import torch.nn as nn
import torch
import pytorch_lightning as pl
from .RNN import RNN
from imu_uwb_pose import config

class trans_model(pl.LightningModule):
    r"""
    Inputs - global orientation and uwb distances, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config:config):
        super().__init__()
        n_input = 9 * len(config.absolute_joint_angles) + len(config.uwb_dists) + len(config.uwb_floor_dists) + 3 * len(config.acceleration_joints) # add back dist above ground here

        n_output = 3

        self.batch_size = config.batch_size
        
        self.model = RNN(n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)

        self.config = config

        self.loss = nn.MSELoss()
        self.lr = config.lr
        self.save_hyperparameters()
        
        self.validation_step_outputs = []  # Store validation outputs manually

    def forward(self, inputs, lens):
        pred_trans, _, _ = self.model(inputs, lens)
        return pred_trans

    def step(self, batch):
        inputs, target_trans, lengths, _ = batch     # ← lengths is a list/array
        lengths = torch.as_tensor(lengths,           # ➊ make it a tensor
                                dtype=torch.long)

        pred = self(inputs, lengths).reshape(-1, self.config.max_sample_length, 3)
        target = target_trans
        lengths_dev = lengths.to(self.config.device)
        # Padding mask: shape (B, T)
        mask = (
            torch.arange(self.config.max_sample_length, device=self.config.device)
                .unsqueeze(0)                      # shape (1, T)
            < lengths_dev.unsqueeze(1)                   # shape (B, 1)
        ).to(self.config.device)

        # MSE over valid timesteps only
        sq_err = (pred - target).pow(2).sum(-1)      # (B, T)
        loss = (sq_err * mask).sum() / mask.sum()

        return loss, pred, target, lengths

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)
        
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, _, _,_ = self.step(batch)
        
        self.log("validation_step_loss", loss.item(), batch_size=self.batch_size)
        self.validation_step_outputs.append({"loss": loss.item()})  # Collect outputs manually
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        loss, pred_pose, target_pose, lengths = self.step(batch)
        
        return {"loss": loss.item(), "pred": pred_pose, "true": target_pose, "lengths": lengths}

    def test_step(self, batch, batch_idx):
        loss, preds, targets,_ = self.step(batch)  # Reuse your existing 'step' logic
        self.log('test_loss', loss, prog_bar=True)
        return {"test_loss": loss}
    
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
