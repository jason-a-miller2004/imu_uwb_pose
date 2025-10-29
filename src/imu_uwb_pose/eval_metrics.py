import imu_uwb_pose.data_extraction as de
import imu_uwb_pose.config as c
from imu_uwb_pose.utils import default_smpl_input, r6d_to_axis_angle
from scipy.spatial.transform import Rotation as R
# import open3d as o3d
import numpy as np
import torch
import time
import pickle

# calculate mean joint angle error
# only for the first 22 joints
def mean_joint_angle_error(pred, gt, lengths, config):
    # get global orientation
    pred = pred.reshape(-1, 66)
    gt = gt.reshape(-1, 66)

    pred_orient = de.extract_angle_amass(pred, config, True).cpu().numpy()
    gt_orient = de.extract_angle_amass(gt, config, True).cpu().numpy()

    pred_orient = pred_orient.reshape(-1, 3)
    gt_orient = gt_orient.reshape(-1, 3)

    # convert to rotation matrix
    pred_rot = R.from_rotvec(pred_orient)
    gt_rot = R.from_rotvec(gt_orient)

    pred_orient = R.as_matrix(pred_rot)
    gt_orient = R.as_matrix(gt_rot)
    
    # get the relative rotation matrix
    pred_rot_matrix = pred_orient.reshape(-1, 3, 3)
    gt_rot_matrix = gt_orient.reshape(-1, 3, 3)

    # Transpose gt matrices
    gt_trans = np.transpose(gt_rot_matrix, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(pred_rot_matrix, gt_trans)
    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r.shape[0]):
        aa = R.from_matrix(r[i, :, :]).as_rotvec()
        angles.append(np.linalg.norm(aa))

    angles = np.array(angles)
    angles = angles.reshape(-1, config.max_sample_length, 22)
    # Initialize accumulator for each of the 22 joints

    # # plot the angle error for left and right joints for valid frames
    # # joints are 7, 8
    # left_ankle = angles[:, :, 7]
    # right_ankle = angles[:, :, 8]

    # left_errors = []
    # right_errors = []
    # for i in range(left_ankle.shape[0]):
    #     left_errors += left_ankle[i, :lengths[i]].flatten().tolist()
    #     right_errors += right_ankle[i, :lengths[i]].flatten().tolist()

    # # plot in matplotlib
    # import matplotlib.pyplot as plt
    # plt.plot(left_errors, label='left ankle')
    # plt.plot(right_errors, label='right ankle')
    # plt.legend()
    # plt.show()

    sums = np.zeros(22, dtype=float)
    
    # Accumulate the sum of angles per joint
    for i in range(angles.shape[0]):
        # Only consider up to lengths[i] frames for sample i
        sums += angles[i, :lengths[i]].sum(axis=0)
    
    # Divide by the total number of frames across all samples
    total_frames = np.sum(lengths)
    means = sums / total_frames

    return means

def mean_joint_and_vertex_error(pred, gt, lengths, config, body_model):
    """
    Computes:
      1) Mean Per Joint Position Error (MPJPE) for the first 22 joints.
      2) Mean Per Joint Vertex Error (MPJVE) for all SMPL vertices.
    
    Both are pelvis-aligned and returned in centimeters (cm).

    :param pred:      np.ndarray of shape (B*max_length, 23, 3) OR (B, max_length, 23, 3) flattened
    :param gt:        same shape as pred
    :param lengths:   list of sequence lengths (integers)
    :param config:    must have 'max_sample_length' (int); also used in default_smpl_input
    :param translation: if True, pred[:, 22, :] is the transl parameter for SMPL
    :return: (mpjpe_cm, mpjve_cm) both floats in centimeters
    """

    # 1) Reshape for SMPL input (if not already flattened)
    pred = pred.reshape(-1, 22, 3)   # => (B*max_length, 23, 3)
    gt   = gt.reshape(-1, 22, 3)

    # 2) Prepare SMPL input
    smpl_input = default_smpl_input(pred.shape[0], config)

    # ----------------- Forward pass for predicted -----------------
    smpl_input['global_orient'] = pred[:, 0, :].to(config.device)      # (B*F, 3)
    smpl_input['body_pose']     = pred[:, 1:22, :].to(config.device)   # (B*F, 21, 3)

    pred_output = body_model(**smpl_input)
    pred_joints = pred_output.joints[:, 0:22, :].cpu().numpy()     # shape (B*F, 22, 3)
    pred_verts  = pred_output.vertices.cpu().numpy()               # shape (B*F, V, 3)

    # ----------------- Forward pass for GT -----------------
    smpl_input['global_orient'] = gt[:, 0, :].to(config.device)
    smpl_input['body_pose']     = gt[:, 1:22, :].to(config.device)

    gt_output   = body_model(**smpl_input)
    gt_joints   = gt_output.joints[:, 0:22, :].cpu().numpy()   # shape (B*F, 22, 3)
    gt_verts    = gt_output.vertices.cpu().numpy()             # shape (B*F, V, 3)

    # 3) Pelvis alignment for both joints & vertices
    #    Pelvis is joint 0 => shape (B*F, 1, 3)
    pred_pelvis = pred_joints[:, 0:1, :]   # => (B*F, 1, 3)
    gt_pelvis   = gt_joints[:, 0:1, :]

    # Subtract pelvis from every joint
    pred_joints_aligned = pred_joints - pred_pelvis  # (B*F, 22, 3)
    gt_joints_aligned   = gt_joints   - gt_pelvis

    # Subtract pelvis from every vertex
    pred_verts_aligned = pred_verts - pred_pelvis
    gt_verts_aligned   = gt_verts   - gt_pelvis

    # 4) Compute L2 errors in meters
    #    MPJPE => shape (B*F, 22)
    joint_diff = np.linalg.norm(pred_joints_aligned - gt_joints_aligned, axis=2)

    #    MPJVE => shape (B*F, V)
    vert_diff = np.linalg.norm(pred_verts_aligned - gt_verts_aligned, axis=2)

    # 5) Reshape to (B, max_length, ...)
    B         = len(lengths)
    max_len   = config.max_sample_length
    V         = vert_diff.shape[1]  # number of vertices
    joint_diff = joint_diff.reshape(B, max_len, 22)
    vert_diff  = vert_diff.reshape(B, max_len, V)

    # 6) Accumulate sums up to each sequence’s length, then compute final average
    sum_joint_diffs = np.zeros(22,)
    sum_vert_diffs = np.zeros(V,)
    total_frames    = 0

    for i in range(B):
        seq_len = lengths[i]

        sum_joint_diffs += joint_diff[i, :seq_len].sum(axis=0) # scalar: mean over (seq_len * 22)
        sum_vert_diffs  += vert_diff[i, :seq_len].sum(axis=0)  # scalar: mean over (seq_len * V)
        total_frames    += seq_len

    # Average in meters
    mpjpe_meters = sum_joint_diffs / float(total_frames)
    mpjve_meters = sum_vert_diffs  / float(total_frames)

    # Convert to centimeters
    mpjpe_cm = 100.0 * mpjpe_meters
    mpjve_cm = 100.0 * mpjve_meters

    return mpjpe_cm, mpjve_cm

def mean_per_joint_jitter(pred, lengths, body_model, config):
    """
    Computes the *average jerk* (3rd derivative) for each of the 22 body joints
    in the SMPL model. Returns an array of shape (22,) with the mean jerk magnitude
    per joint over all frames in all sequences.
    
    If you want one global scalar, you can do .mean() on the result.
    """
    # 1) Reshape so we can feed SMPL
    #    Suppose pred originally (B, max_length, 23, 3) => flatten => (B*max_length, 23, 3)
    pred = pred.reshape(-1, 22, 3)

    # 2) SMPL forward pass
    smpl_input = default_smpl_input(pred.shape[0], config)
    smpl_input['global_orient'] = pred[:, 0, :].to(config.device)
    smpl_input['body_pose']     = pred[:, 1:22, :].to(config.device)

    pred_output = body_model(**smpl_input)
    # shape => (B*max_length, 49, 3) or so; we'll slice 0:22
    pred_joints_all = pred_output.joints[:, 0:22, :].cpu().numpy()  # => (B*max_length, 22, 3)

    # 3) Reshape back => (B, max_length, 22, 3) so we can handle sequences individually
    B         = len(lengths)
    max_len   = config.max_sample_length
    pred_joints_all = pred_joints_all.reshape(B, max_len, 22, 3)

    # We'll accumulate sum of jerk magnitudes for each joint, across all frames:
    per_joint_sum = np.zeros((22,), dtype=np.float64)
    total_jerk_frames = 0

    for i in range(B):
        seq_len = lengths[i]
        # For jerk, we need at least 4 frames => positions up to seq_len
        if seq_len < 4:
            continue

        # shape => (seq_len, 22, 3)
        seq_joints = pred_joints_all[i, :seq_len, :, :]

        # velocity => (seq_len-1, 22, 3)
        vel = seq_joints[1:] - seq_joints[:-1]
        # acceleration => (seq_len-2, 22, 3)
        acc = vel[1:] - vel[:-1]
        # jerk => (seq_len-3, 22, 3)
        jer = acc[1:] - acc[:-1]

        # Euclidean norm => (seq_len-3, 22)
        jer_norm = np.linalg.norm(jer, axis=2)

        # sum across all time frames => shape (22,)
        per_joint_sum += jer_norm.sum(axis=0)
        total_jerk_frames += (seq_len - 3)

    # Mean jerk magnitude for each joint => (22,)
    if total_jerk_frames > 0:
        per_joint_jitter = per_joint_sum / float(total_jerk_frames)
    else:
        per_joint_jitter = np.zeros((22,), dtype=np.float64)

    return per_joint_jitter

def mean_trans_sec_err(pred, target, lengths):
    sec_errs = []
    f = 30
    for i in range(len(lengths)):
        if (lengths[i] != 150):
            continue
        joint_p = pred[i]
        joint_t = target[i]
        te = ((joint_p[f:,] - joint_p[:-f,]) - (joint_t[f:,] - joint_t[:-f,])).norm(dim=1)*100    # N, 1

        sec_errs.append(te)

    return np.asarray(sec_errs)

def mean_drift_err(pred, target, lengths):
    f = 30
    for i in range(len(lengths)):
        if (lengths[i] != 150):
            continue
        joint_p = pred[i]
        joint_t = target[i]
        drift = ((joint_p - joint_p[0,:]) - (joint_t - joint_t[0,:])).norm(dim=1) * 100
        print(f'drift {drift}')
def get_metrics(outputs, smpl, config):
    angle_error = np.zeros(22,)
    joint_error = np.zeros(22,)
    vertex_error = np.zeros(10475,)
    jitter = np.zeros(22,)

    for i in range(len(outputs)):
        # convert pred and true to axis angle
        pred_r6d = outputs[i]['pred'].reshape(-1, 6)
        true_r6d = outputs[i]['true'].reshape(-1, 6)

        pred_aa = torch.tensor(r6d_to_axis_angle(pred_r6d), dtype=torch.float32).to(config.device)
        true_aa = torch.tensor(r6d_to_axis_angle(true_r6d), dtype=torch.float32).to(config.device)

        pred = pred_aa.reshape(-1, config.max_sample_length, 22, 3)
        true = true_aa.reshape(-1, config.max_sample_length, 22, 3)

        lengths = outputs[i]['lengths']
        print(f'processing output {i} pred shape {pred.shape} true shape {true.shape}')

        mean_angle_error = mean_joint_angle_error(pred, true, lengths, config)
        angle_error += mean_angle_error
        
        mean_joint_error, mean_vertex_error = mean_joint_and_vertex_error(pred, true, lengths, config, smpl)
        joint_error += mean_joint_error
        vertex_error += mean_vertex_error

        mean_jitter = mean_per_joint_jitter(pred, lengths, smpl, config)
        jitter += mean_jitter


    angle_error = angle_error / len(outputs)
    joint_error = joint_error / len(outputs)
    vertex_error = vertex_error / len(outputs)
    jitter = jitter / len(outputs)
    
    return {
        'angle_error': angle_error,
        'joint_error': joint_error,
        'vertex_error': vertex_error,
        'jitter': jitter
    }

def get_trans_metrics(outputs):
    avg_err = 0
    sec_errs = []
    for i in range(len(outputs)):
        pred = outputs[i]['pred']
        target = outputs[i]['true']
        lengths = outputs[i]['lengths']

        err = mean_trans_sec_err(pred, target, lengths)
        mean_drift_err(pred, target, lengths)
        avg_err += np.mean(err)
        sec_errs.append(err)
    return avg_err / len(outputs)