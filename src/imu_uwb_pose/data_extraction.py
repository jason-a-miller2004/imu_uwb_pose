import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from imu_uwb_pose import utils
import smplx
from matplotlib import pyplot as plt
import time

def extract_amass(cdata, config):
    """
    Extract the data from the AMASS dataset
    """

    if 'mocap_framerate' not in cdata:
        print('does not contain mocap_framerate')
        return None

    framerate = int(cdata['mocap_framerate'])
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

    body_parms = {
        'global_orient': torch.Tensor(pose[:, :3]).to(config.device), # controls the global root orientation
        'body_pose': torch.Tensor(pose[:, 3:66]).to(config.device), # controls the body
        'transl': torch.Tensor(tran).to(config.device), # controls the global body position
    }

    smpl_params = utils.default_smpl_input(pose.shape[0], config)

    for key in body_parms.keys():
        smpl_params[key] = body_parms[key]

    smpl = smplx.create(config.body_model, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         batch_size=1,
                         ext='npz',
                         age='adult').to(config.device)
    
    output = smpl(**{k: v for k, v in smpl_params.items()})
    
    # visualize output
    # uncomment if you want to see visualization of movement
    # vertices = output.vertices.detach().cpu().numpy()
    # faces = smpl.faces
    # visualize_smplx_mesh(vertices, faces)

    uwb_distances = extract_uwb_amass(output, config)
    print(f'UWB distances shape: {uwb_distances.shape}')
    angles = extract_angle_amass(pose, config)
    print(f'Angles shape: {angles.shape}')
    floor_distances = extract_dist_floor_amass(output, config)
    print(f'Floor distances shape: {floor_distances.shape}')

    # Reshape angles from (frames, num_angles, 3) to (frames, num_angles * 3)
    angles_reshaped = angles.reshape(angles.shape[0], -1)

    # concat so that the shape is (frames, (angle1, angle2, ..., uwb dist 1, uwb dist 2, ..., uwb1 to floor1, uwb2 to floor2))
    combined_features = torch.cat([angles_reshaped, uwb_distances, floor_distances], dim=1)
    print(f'Combined features shape: {combined_features.shape}')

    params = torch.cat([body_parms['global_orient'], body_parms['body_pose'], body_parms['transl']], dim=1)

    print(f'Params shape: {params.shape}')
    print(f'Joints shape: {output.joints.shape}')
    print()
    return {
        'x': combined_features.detach().cpu().type(torch.float32),
        'y': params.detach().cpu().type(torch.float32),
        'joints': output.joints.detach().cpu().type(torch.float32)[:, 0:22, :],
    }



def extract_angle_amass(pose, config):
    """
    Extract the angles from the AMASS dataset
    """
    axis_angle_pose = pose[:,0:66].reshape(-1, 3)
    
    # only keep body components
    rotation = R.from_rotvec(axis_angle_pose)

    # convert relative joint angles to rotation matrix representation
    r_matrix = torch.tensor(rotation.as_matrix()).to(config.device)

    # get the parent array
    parent = utils.get_parent_array(config.get_smpl_skeleton(), config)
    # calculate the global angle
    global_rot = utils.forward_kinematics_R(r_matrix, parent)

    absolute_joints = config.absolute_joint_angles
    selected_rotations = global_rot[:, absolute_joints, :, :]

    selected_rotations = selected_rotations.reshape(-1,3,3)
    # convert back to axis-angle representation
    selected_rotations = R.from_matrix(selected_rotations.cpu().numpy())
    selected_rotations = selected_rotations.as_rotvec()

    selected_rotations = torch.tensor(selected_rotations.reshape(-1, len(absolute_joints), 3), device=config.device)

    # Convert to numpy for visualization
    selected_rotations_np = selected_rotations.cpu().numpy()

    # Plot each joint's three rotation components
    num_joints = len(absolute_joints)
    time_steps = selected_rotations_np.shape[0]

    # for joint_idx in range(num_joints):
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(range(time_steps), selected_rotations_np[:, joint_idx, 0], label="X-axis")
    #     plt.plot(range(time_steps), selected_rotations_np[:, joint_idx, 1], label="Y-axis")
    #     plt.plot(range(time_steps), selected_rotations_np[:, joint_idx, 2], label="Z-axis")

    #     plt.xlabel("Time Step")
    #     plt.ylabel("Rotation (radians)")
    #     plt.title(f"Joint {absolute_joints[joint_idx]} Rotation Over Time")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    return selected_rotations

def extract_uwb_amass(output, config):
    """
    Extract the UWB data from the AMASS dataset
    """
    joints = output.joints  # Shape: (frames, 127, 3)

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

def extract_dist_floor_amass(output, config):
    """
    Extract the distance from the floor data from the AMASS dataset
    """
    joints = output.joints  # Shape: (frames, 127, 3)

    # Get the root joint position
    root_position = joints[:, config.uwb_floor_dists, :]  # Shape: (frames, 3)

    # Compute the distance from the floor: z-coordinate of the root joint
    dist_floor = root_position[:, :, 2]  # Shape: (frames,)

    # uncomment if you want to see distances to floor plotted for validity
    # plt.plot(dist_floor.detach().cpu().numpy()[:, 0])
    # plt.plot(dist_floor.detach().cpu().numpy()[:, 1])
    # plt.show()

    return dist_floor  # Torch tensor of shape (frames,)