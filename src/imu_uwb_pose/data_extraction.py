import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from imu_uwb_pose import utils
import smplx
from matplotlib import pyplot as plt
import open3d as o3d
import time

def extract(cdata, config):
    """
    Extract the data from the AMASS dataset
    """
    if 'mocap_frame_rate' not in cdata:
        return None

    framerate = int(cdata['mocap_frame_rate'])
    if framerate == 120: step = 4
    elif framerate == 60 or framerate == 59: step = 2
    elif framerate == 30: step = 1
    else: return None

    pose = cdata['poses'][::step].astype(np.float32)
    tran = cdata['trans'][::step].astype(np.float32)
    gender = cdata['gender'].item()

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

    time_length = pose.shape[0]
    num_betas = 10 # number of body parameters

    body_parms = {
        'global_orient': torch.Tensor(pose[:, :3]).to(config.device), # controls the global root orientation
        'body_pose': torch.Tensor(pose[:, 3:66]).to(config.device), # controls the body
        'transl': torch.Tensor(tran).to(config.device), # controls the global body position
        'betas': torch.Tensor(np.repeat(cdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(config.device), # controls the body shape. Body shape is static
    }

    smpl = smplx.create(config.body_model, model_type='smplx',
                         gender=gender, use_face_contour=False,
                         num_betas=10,
                         batch_size=time_length,
                         ext='npz',
                         age='adult').to(config.device)
    
    output = smpl(**{k: v for k, v in body_parms.items()})
    
    # visualize output
    vertices = output.vertices.detach().cpu().numpy()
    faces = smpl.faces

    # uncomment if you want to see visualization of movement
    # visualize_smplx_mesh(vertices, faces)

    uwb_distances = extract_uwb(output, config)
    print(f'UWB distances shape: {uwb_distances.shape}')
    angles = extract_angle(pose, config)
    print(f'Angles shape: {angles.shape}')
    floor_distances = extract_dist_floor(output, config)
    print(f'Floor distances shape: {floor_distances.shape}')

    # Reshape angles from (frames, num_angles, 3) to (frames, num_angles * 3)
    angles_reshaped = angles.reshape(angles.shape[0], -1)

    # concat so that the shape is (frames, (angle1, angle2, ..., uwb dist 1, uwb dist 2, ..., uwb1 to floor1, uwb2 to floor2))
    combined_features = torch.cat([angles_reshaped, uwb_distances, floor_distances], dim=1)
    print(f'Combined features shape: {combined_features.shape}')

    return combined_features



def extract_angle(pose, config):
    """
    Extract the angles from the AMASS dataset
    """
    axis_angle_pose = pose[:,0:72].reshape(-1, 3)
    
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
    selected_rotations = R.from_matrix(selected_rotations)
    selected_rotations = selected_rotations.as_rotvec()

    selected_rotations = torch.tensor(selected_rotations.reshape(-1, len(absolute_joints), 3), device=config.device)

    # Convert to numpy for visualization
    selected_rotations_np = selected_rotations.cpu().numpy()

    # Plot each joint's three rotation components
    num_joints = len(absolute_joints)
    time_steps = selected_rotations_np.shape[0]

    for joint_idx in range(num_joints):
        plt.figure(figsize=(8, 5))
        plt.plot(range(time_steps), selected_rotations_np[:, joint_idx, 0], label="X-axis")
        plt.plot(range(time_steps), selected_rotations_np[:, joint_idx, 1], label="Y-axis")
        plt.plot(range(time_steps), selected_rotations_np[:, joint_idx, 2], label="Z-axis")

        plt.xlabel("Time Step")
        plt.ylabel("Rotation (radians)")
        plt.title(f"Joint {absolute_joints[joint_idx]} Rotation Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    return selected_rotations

def extract_uwb(output, config):
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

    plt.plot(uwb_distances.detach().cpu().numpy()[:, 0])
    plt.title("UWB Distance Over Time")

    return uwb_distances  # Torch tensor of shape (frames, len(uwb_dists))

def extract_dist_floor(output, config):
    """
    Extract the distance from the floor data from the AMASS dataset
    """
    joints = output.joints  # Shape: (frames, 127, 3)

    # Get the root joint position
    root_position = joints[:, config.uwb_floor_dists, :]  # Shape: (frames, 3)

    # Compute the distance from the floor: z-coordinate of the root joint
    dist_floor = root_position[:, :, 2]  # Shape: (frames,)

    # uncomment if you want to see distances to floor plotted for validity
    plt.plot(dist_floor.detach().cpu().numpy()[:, 0])
    plt.plot(dist_floor.detach().cpu().numpy()[:, 1])
    plt.show()

    return dist_floor  # Torch tensor of shape (frames,)


def visualize_smplx_mesh(vertices_sequence, faces, delay=0.05):
    """
    Visualizes an SMPL-X mesh sequence with a dynamically positioned ground plane using Open3D.

    Args:
        vertices_sequence: np.ndarray of shape (num_frames, num_vertices, 3),
                           containing 3D vertex positions over time.
        faces: np.ndarray of shape (num_faces, 3), defining triangular faces.
        plane_size: Float, size of the ground plane.
        delay: Time delay between frames in seconds.
    """
    num_frames, num_vertices, _ = vertices_sequence.shape

    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # show coordinate axes
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])

    # Create and add coordinate frame object
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # Create mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_sequence[0])  # Initial frame vertices
    mesh.triangles = o3d.utility.Vector3iVector(faces)  # Mesh connectivity
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.6, 0.6, 0.6])  # Light gray mesh
    vis.add_geometry(mesh)

    # Animation loop
    for frame in range(num_frames):
        mesh.vertices = o3d.utility.Vector3dVector(vertices_sequence[frame])  # Update vertex positions
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)
        
        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)

    vis.destroy_window()
    