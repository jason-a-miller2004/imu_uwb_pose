import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from imu_uwb_pose import utils
import smplx
from matplotlib import pyplot as plt
import time
from imu_uwb_pose import config as c, utils as u
import os
import pickle

def extract_footposer_mocap(action_path, config, smpl, skiprate=1):
    files = os.listdir(action_path)

    # extract the align.csv
    align_file = [f for f in files if f.endswith("align.csv")][0]
    align_path = os.path.join(action_path, align_file)
    print("align path", align_path)
    data = np.loadtxt(align_path, delimiter=',')
    smpl_start, smpl_finish = int(data[0]), int(data[1])
    mocap_file = [f for f in files if f.endswith("stageii.pkl")][0]
    mocap_path = os.path.join(action_path, mocap_file)

    cdata = np.load(mocap_path, allow_pickle=True)
    pose = cdata['fullpose'].astype(np.float32)
    tran = cdata['trans'][::skiprate].astype(np.float32)


    # currently at 120hz resample to 30hz
    pose = torch.tensor(pose[::skiprate, :66])
    tran = torch.tensor(tran)[smpl_start:smpl_finish,:]

    pose = pose[smpl_start:smpl_finish, :]

    print(f'pose shape {pose.shape}')
    vertices,joints,faces = u.get_smpl_output(smpl, pose, config)
    _,trans_joints,_ = u.get_smpl_output(smpl, pose,config, tran)
    # angles
    angles = extract_angle_amass(pose, config).reshape(-1, 6 * len(config.absolute_joint_angles))
    uwb_dists = extract_uwb_amass(joints, config).reshape(-1, len(config.uwb_dists))
    floor_dists = extract_dist_floor_amass(trans_joints, config).reshape(-1, len(config.uwb_floor_dists))

    pose = pose.reshape(-1, 3)
    r6d_pose = u.axis_angle_to_r6d(pose)
    r6d_pose = r6d_pose.reshape(-1, 22, 6)

    x = torch.concatenate((angles, uwb_dists, floor_dists),dim=1)
    return {
        'x': x.detach().cpu().type(torch.float32),
        'y': r6d_pose.detach().cpu().type(torch.float32),
        'joints': joints.detach().cpu().type(torch.float32)[:, 0:22, :],
        'trans': torch.tensor(cdata['trans'])
    }

def extract_footposer(action_path, config, smpl, skiprate=1):
    files = os.listdir(action_path)

    # extract the align.csv
    align_file = [f for f in files if f.endswith("align.csv")][0]
    align_path = os.path.join(action_path, align_file)
    print("align path", align_path)
    data = np.loadtxt(align_path, delimiter=',')
    smpl_start, smpl_finish, sensor_start, sensor_finish = int(data[0]), int(data[1]), int(data[2]), int(data[3])

    # extract the sensor data and format it
    sensor_file = [f for f in files if f.endswith("filtered.pkl")][0]
    sensor_path = os.path.join(action_path, sensor_file)
    with open(sensor_path, 'rb') as file:
        sensor_data = pickle.load(file)

    # combine in order left_imu, right_imu, uwb_dists
    left_imu = u.rotation_matrix_to_r6d(torch.tensor(sensor_data['left_imu']))[sensor_start:sensor_finish, :]
    right_imu = u.rotation_matrix_to_r6d(torch.tensor(sensor_data['right_imu']))[sensor_start:sensor_finish, :]
    uwb_dists = sensor_data['uwb'][sensor_start:sensor_finish, :]
    left_altitude = sensor_data['left_altitude_filtered'][:, np.newaxis]
    right_altitude = sensor_data['right_altitude_filtered'][:, np.newaxis]
    # concat all the fields into left imu, right imu, uwb dists
    sensor_data = np.concatenate([left_imu, right_imu, uwb_dists, left_altitude, right_altitude], axis=1)
    # convert to torch tensor
    sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
    print(f'sensor data shape {sensor_data.shape}')

    # load the mocap data
    mocap_file = [f for f in files if f.endswith("stageii.pkl")][0]
    mocap_path = os.path.join(action_path, mocap_file)

    cdata = np.load(mocap_path, allow_pickle=True)
    pose = cdata['fullpose'].astype(np.float32)

    # currently at 120hz resample to 30hz
    pose = torch.tensor(pose[::skiprate, :66])
    pose = pose[smpl_start:smpl_finish, :]

    print(f'pose shape {pose.shape}')
    vertices,joints,faces = u.get_smpl_output(smpl, pose, config)
            
    # get the pose in r6d
    pose = pose.reshape(-1, 3)
    r6d_pose = u.axis_angle_to_r6d(pose)
    r6d_pose = r6d_pose.reshape(-1, 22, 6)
    return {
        'x': sensor_data.detach().cpu().type(torch.float32),
        'y': r6d_pose.detach().cpu().type(torch.float32),
        'joints': joints.detach().cpu().type(torch.float32)[:, 0:22, :],
        'trans': torch.tensor(cdata['trans'])[smpl_start:smpl_finish]
    }

def extract_amass(cdata, smpl, config):
    """
    Extract the data from the AMASS dataset
    """
    if 'mocap_framerate' in cdata:
        framerate = int(cdata['mocap_framerate'])
    elif 'mocap_frame_rate' in cdata:
        framerate = int(cdata['mocap_frame_rate'])
    else:
        print('does not contain mocap_framerate')
        return None

    print(f'framerate {framerate}')
    if framerate == 120: step = 4
    elif framerate == 60 or framerate == 59: step = 2
    elif framerate == 30: step = 1
    else: return None

    pose = cdata['poses'][::step].astype(np.float32)
    tran = cdata['trans'][::step].astype(np.float32)

    # align AMASS global frame with a different orientation
    # right now set to no rotation
    amass_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    tran = np.dot(tran, amass_rot.T)

    rotvec = pose[:, :3]  # axis-angle vectors for each sample

    # Create a SciPy Rotation object from the axis-angle vectors.
    rot = R.from_rotvec(rotvec)
    # Convert to rotation matrices (shape: (B, 3, 3))
    rot_matrices = rot.as_matrix()

    # Apply the fixed rotation (amass_rot) to each rotation matrix.
    # This multiplies amass_rot with each sample's rotation matrix.
    aligned_rot_matrices = np.einsum('ij,bjk->bik', amass_rot, rot_matrices)

    # Convert the aligned rotation matrices back to axis–angle representation.
    aligned_rot = R.from_matrix(aligned_rot_matrices)
    aligned_rotvec = aligned_rot.as_rotvec()  # shape: (B, 3)

    # Update pose with the new aligned axis-angle rotations.
    # (If needed, you can also send the tensor to the appropriate device.)
    pose[:, :3] = aligned_rotvec
    pose = torch.tensor(pose, dtype=torch.float32)
    tran = torch.tensor(tran, dtype=torch.float32)
    
    vertices, joints, faces = utils.get_smpl_output(smpl, pose, config)
    
    # visualize output
    # uncomment if you want to see visualization of movement
    # vertices = output.vertices.detach().cpu().numpy()
    # faces = smpl.faces
    # visualize_smplx_mesh(vertices, faces)

    uwb_distances = extract_uwb_amass(joints, config)
    print(f'UWB distances shape: {uwb_distances.shape}')
    angles = extract_angle_amass(pose, config)
    print(f'Angles shape: {angles.shape}')

    # get smpl with translation to extract floor dists
    _, trans_joints, _ = utils.get_smpl_output(smpl, pose, config, tran)
    floor_distances = extract_dist_floor_amass(trans_joints, config)
    # print(f'Floor distances shape: {floor_distances.shape}')

    # extract acceleration
    accel = extract_acceleration_amass(joints,config)
    accel = accel.reshape(-1, 6)

    # convert angles from axis-angle to r6d
    num_angles = angles.shape[1]
    angles = angles.reshape(-1, 3)
    angles_rot = R.from_rotvec(angles)
    angles_matrix = R.as_matrix(angles_rot)
    # angles = utils.axis_angle_to_r6d(angles)
    angles = torch.tensor(angles_matrix.reshape(-1, num_angles, 9))

    print(f'Angles shape after conversion: {angles.shape}')

    # Reshape angles from (frames, num_angles, 9) to (frames, num_angles * 9)
    angles_reshaped = angles.reshape(angles.shape[0], -1)

    # concat so that the shape is (frames, (angle1, angle2, ..., uwb dist 1, uwb dist 2, ..., uwb1 to floor1, uwb2 to floor2))
    combined_features = torch.cat([angles_reshaped, uwb_distances, floor_distances, accel], dim=1)
    print(f'Combined features shape: {combined_features.shape}')

    # convert global orient and body pose to r6d
    global_r6d = utils.axis_angle_to_r6d(pose[:,:3]).reshape(-1, 1, 6)

    body_pose = pose[:,3:66].reshape(-1, 3)
    body_r6d = utils.axis_angle_to_r6d(body_pose)
    body_r6d = body_r6d.reshape(-1, 21, 6)
    params = torch.cat([global_r6d, body_r6d], dim=1)

    # get deltaTrans by subtracting initial translation
    deltaTran = tran - tran[0]

    print(f'Params shape: {params.shape}')
    print(f'Joints shape: {joints.shape}')
    print()
    return {
        'x': combined_features.detach().cpu().type(torch.float32),
        'y': params.detach().cpu().type(torch.float32),
        'joints': joints.detach().cpu().type(torch.float32)[:, 0:22, :],
        'trans': deltaTran
    }



def extract_angle_amass(pose, config, all=False):
    """
    Extract the angles from the AMASS dataset
    """
    axis_angle_pose = pose[:,0:66].reshape(-1, 3)
    r_matrix = u.axis_angle_to_rotation_matrix(axis_angle_pose)
    # get the parent array
    parent = utils.get_parent_array(config.get_smpl_skeleton(), config)
    # calculate the global angle
    global_rot = utils.forward_kinematics_R(r_matrix, parent)

    if (not all):
        absolute_joints = config.absolute_joint_angles
        selected_rotations = global_rot[:, absolute_joints, :, :]
        num_joints = len(absolute_joints)
    else:
        selected_rotations = global_rot[:, :22, :, :]
        num_joints = 22

    selected_rotations = selected_rotations.reshape(-1,3,3)
    r6d_angles = u.rotation_matrix_to_r6d(selected_rotations)
    return r6d_angles

def extract_uwb_amass(joints, config):
    """
    Extract the UWB data from the AMASS dataset
    """

    p1_indices = torch.tensor([p1 for p1, p2 in config.uwb_dists], device=joints.device)
    p2_indices = torch.tensor([p2 for p1, p2 in config.uwb_dists], device=joints.device)

    # Gather joint positions
    p1_positions = joints[:, p1_indices, :]  # Shape: (frames, num_dists, 3)
    p2_positions = joints[:, p2_indices, :]  # Shape: (frames, num_dists, 3)

    # Compute Euclidean distance: ||p1 - p2||_2
    uwb_distances = torch.norm(p1_positions - p2_positions, dim=2)  # Shape: (frames, num_dists)

    # plt.plot(uwb_distances.detach().cpu().numpy()[:, 0])
    # plt.title("UWB Distance Over Time")

    return uwb_distances  # Torch tensor of shape (frames, len(uwb_dists))

def extract_acceleration_amass(joints, config):
    pos_vecs_idces = torch.tensor(config.acceleration_joints)

    pos = joints[:, pos_vecs_idces, :]

        # Δt in seconds between frames
    dt = 1.0 / float(30)

    # First-order finite differences → velocity  (T-1, M, 3)
    vel = (pos[1:] - pos[:-1]) / dt

    # Second-order finite differences → acceleration  (T-2, M, 3)
    acc = (vel[1:] - vel[:-1]) / dt

    # Pad to original length: prepend and append zeros
    acc = torch.nn.functional.pad(acc,   # pad last dim first → (0,0),
                                  pad=(0, 0, 0, 0, 1, 1),  # pad time dim → (1,1)
                                  mode="constant", value=0.0)

    return acc
    
def extract_dist_floor_amass(joints, config):
    """
    Extract the distance from the floor data from the AMASS dataset
    """

    # Get the root joint position
    root_position = joints[:, config.uwb_floor_dists, :]  # Shape: (frames, 3)

    # Compute the distance from the floor: z-coordinate of the root joint
    dist_floor = root_position[:, :, 2]  # Shape: (frames,)

    # uncomment if you want to see distances to floor plotted for validity
    # plt.plot(dist_floor.detach().cpu().numpy()[:, 0])
    # plt.plot(dist_floor.detach().cpu().numpy()[:, 1])
    # plt.show()

    return dist_floor  # Torch tensor of shape (frames,)