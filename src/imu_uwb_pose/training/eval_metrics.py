import imu_uwb_pose.data_extraction as de
import imu_uwb_pose.config as c
from imu_uwb_pose.utils import default_smpl_input, r6d_to_axis_angle
from scipy.spatial.transform import Rotation as R
# import open3d as o3d
import numpy as np
# import open3d as o3d
import torch
import time
import smplx

config = c.config()

pred = torch.load("pred.pt").reshape(-1, config.max_sample_length, 22, 6)
true = torch.load("true.pt").reshape(-1, config.max_sample_length, 22, 6)
lengths = torch.load("lengths.pt")

body_model = smplx.create(config.body_model, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         batch_size=1,
                         ext='npz',
                         age='adult').to(config.device)

# calculate mean joint angle error
# only for the first 22 joints
def mean_joint_angle_error(pred, gt, lengths, config):
    # get global orientation
    pred = pred.reshape(-1, 66)
    gt = gt.reshape(-1, 66)

    pred_orient = de.extract_angle(pred, config, True).cpu().numpy()
    gt_orient = de.extract_angle(gt, config, True).cpu().numpy()

    print('pred shape ', pred_orient.shape)
    print('gt shape ', gt_orient.shape)

    # get the relative rotation matrix
    pred_rot_matrix = pred_orient.reshape(-1, 3, 3)
    gt_rot_matrix = gt_orient.reshape(-1, 3, 3)

    # Transpose gt matrices
    gt_trans = np.transpose(gt_rot_matrix, [0, 2, 1])
    print('gt trans shape ', gt_trans.shape)

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

    # plot the angle error for left and right joints for valid frames
    # joints are 7, 8
    left_ankle = angles[:, :, 7]
    right_ankle = angles[:, :, 8]

    left_errors = []
    right_errors = []
    for i in range(left_ankle.shape[0]):
        left_errors += left_ankle[i, :lengths[i]].flatten().tolist()
        right_errors += right_ankle[i, :lengths[i]].flatten().tolist()

    # plot in matplotlib
    import matplotlib.pyplot as plt
    plt.plot(left_errors, label='left ankle')
    plt.plot(right_errors, label='right ankle')
    plt.legend()
    plt.show()

    sums = np.zeros(22, dtype=float)
    
    # Accumulate the sum of angles per joint
    for i in range(angles.shape[0]):
        # Only consider up to lengths[i] frames for sample i
        sums += angles[i, :lengths[i]].sum(axis=0)
    
    # Divide by the total number of frames across all samples
    total_frames = np.sum(lengths)
    means = sums / total_frames

    print(f'means shape ', means.shape)
    print(means)
    return np.mean(means)

def mean_joint_and_vertex_error(pred, gt, lengths, config, translation=True):
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
    pred = pred.reshape(-1, 23, 3)   # => (B*max_length, 23, 3)
    gt   = gt.reshape(-1, 23, 3)

    # 2) Prepare SMPL input
    smpl_input = default_smpl_input(pred.shape[0], config)

    # ----------------- Forward pass for predicted -----------------
    smpl_input['global_orient'] = pred[:, 0, :]      # (B*F, 3)
    smpl_input['body_pose']     = pred[:, 1:22, :]   # (B*F, 21, 3)
    if translation: 
        smpl_input['transl'] = pred[:, 22, :]        # (B*F, 3)

    pred_output = body_model(**smpl_input)
    pred_joints = pred_output.joints[:, 0:22, :]     # shape (B*F, 22, 3)
    pred_verts  = pred_output.vertices               # shape (B*F, V, 3)

    # ----------------- Forward pass for GT -----------------
    smpl_input['global_orient'] = gt[:, 0, :]
    smpl_input['body_pose']     = gt[:, 1:22, :]
    if translation:
        smpl_input['transl'] = gt[:, 22, :]

    gt_output   = body_model(**smpl_input)
    gt_joints   = gt_output.joints[:, 0:22, :]       # shape (B*F, 22, 3)
    gt_verts    = gt_output.vertices                 # shape (B*F, V, 3)

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
    total_joint_sum = 0.0
    total_vert_sum  = 0.0
    total_frames    = 0

    for i in range(B):
        seq_len = lengths[i]
        # For each frame up to seq_len:
        # - average over 22 joints
        # - average over V vertices
        # Then accumulate.
        seq_joints_mean = joint_diff[i, :seq_len].mean()  # scalar: mean over (seq_len * 22)
        seq_verts_mean  = vert_diff[i, :seq_len].mean()   # scalar: mean over (seq_len * V)

        total_joint_sum += seq_joints_mean * seq_len
        total_vert_sum  += seq_verts_mean  * seq_len
        total_frames    += seq_len

    # Average in meters
    mpjpe_meters = total_joint_sum / float(total_frames)
    mpjve_meters = total_vert_sum  / float(total_frames)

    # Convert to centimeters
    mpjpe_cm = 100.0 * mpjpe_meters
    mpjve_cm = 100.0 * mpjve_meters

    return mpjpe_cm, mpjve_cm

def mean_per_joint_jitter(pred, lengths, config, translation=True):
    """
    Computes the *average jerk* (3rd derivative) for each of the 22 body joints
    in the SMPL model. Returns an array of shape (22,) with the mean jerk magnitude
    per joint over all frames in all sequences.
    
    If you want one global scalar, you can do .mean() on the result.
    """
    # 1) Reshape so we can feed SMPL
    #    Suppose pred originally (B, max_length, 23, 3) => flatten => (B*max_length, 23, 3)
    pred = pred.reshape(-1, 23, 3)

    # 2) SMPL forward pass
    smpl_input = default_smpl_input(pred.shape[0], config)
    smpl_input['global_orient'] = pred[:, 0, :]
    smpl_input['body_pose']     = pred[:, 1:22, :]
    if translation:
        smpl_input['transl'] = pred[:, 22, :]

    pred_output = body_model(**smpl_input)
    # shape => (B*max_length, 49, 3) or so; we'll slice 0:22
    pred_joints_all = pred_output.joints[:, 0:22, :]  # => (B*max_length, 22, 3)

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

    return np.mean(per_joint_jitter)

# def visualize(pred):
#         smpl_input = default_smpl_input(pred.shape[0], config)

#         smpl_input['global_orient'] = pred[:, :3]
#         smpl_input['body_pose'] = pred[:, 3:66]
#         smpl_input['transl'] = pred[:, 66:]

#         output = body_model(**smpl_input)
#         vertices = output.vertices.detach().cpu().numpy()
#         faces = body_model.faces

#         visualize_frames_open3d(vertices, faces, fps=30)

# def visualize_frames_open3d(frames, faces, fps=120):
#     """
#     Visualize a list of (N, 3) point sets in an Open3D window as a mesh,
#     refreshing at the specified fps (default 120Hz).

#     Args:
#         frames (list or array-like): A sequence of arrays, each of shape (N, 3).
#                                      Each array in 'frames' represents the vertex
#                                      positions for that frame.
#         faces (array-like): An array of shape (M, 3), each row containing the vertex
#                             indices for one triangular face.
#         fps (int): Frames per second to update the visualization.
#     """

#     # Create a TriangleMesh geometry
#     mesh = o3d.geometry.TriangleMesh()

#     # Create a Visualizer window
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name='Open3D Mesh', width=1280, height=720)

#     # Optionally add a coordinate frame (useful for reference)
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.5, origin=[0, 0, 0]
#     )
#     vis.add_geometry(coordinate_frame)

#     # Add the initial mesh to the visualizer
#     vis.add_geometry(mesh)

#     # Calculate the time interval between frames
#     frame_interval = 1.0 / fps

#     mesh.triangles = o3d.utility.Vector3iVector(faces)

#     for i, frame_data in enumerate(frames):
#         # Update the mesh's vertices
#         mesh.vertices = o3d.utility.Vector3dVector(frame_data)
#         mesh.compute_vertex_normals()

#         # Update geometry in the visualizer
#         vis.update_geometry(mesh)
#         vis.poll_events()

#         # Optionally reset the viewpoint on the first frame
#         if i == 0:
#             vis.reset_view_point(True)

#         vis.update_renderer()

#         # Wait briefly to maintain your desired fps
#         time.sleep(frame_interval)

#     # Once done, close the window
#     vis.destroy_window()


# convert pred and true to axis angle
pred = r6d_to_axis_angle(pred.reshape(-1, 6)).reshape(-1, config.max_sample_length, 22, 3)
true = r6d_to_axis_angle(true.reshape(-1, 6)).reshape(-1, config.max_sample_length, 22, 3)

print('pred shape ', pred.shape)
print('true shape ', true.shape)
print('mean joint angle error ', mean_joint_angle_error(pred[:, :, :22, :], true[:, :, :22, :], lengths, config))
# joint_error, vertex_error = mean_joint_and_vertex_error(pred, true, lengths, config, translation=True)
# print('joint error ', joint_error)
# print('vertex error ', vertex_error)
# print('mean per joint jitter ', mean_per_joint_jitter(pred, lengths, config, translation=True))