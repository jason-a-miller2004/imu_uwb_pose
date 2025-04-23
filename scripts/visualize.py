import torch
import numpy as np
import smplx
from imu_uwb_pose import config as c, utils
import open3d as o3d
import time
import argparse

# visualize an output file
def visualize_model_output(file_loc, smpl, config):
    results = torch.load(file_loc)
    vertices = []
    faces = None
    for (input, _) in results:
        input = input.reshape(-1, 66)
        smpl_params = utils.default_smpl_input(input.shape[0], config)

        body_parms = {
            'global_orient': torch.Tensor(input[:, :3]).to(config.device), # controls the global root orientation
            'body_pose': torch.Tensor(input[:, 3:66]).to(config.device), # controls the body
        }

        for key in body_parms.keys():
            smpl_params[key] = body_parms[key]

        output = smpl(**{k: v for k, v in smpl_params.items()})
        vertices.append(output.vertices.detach().cpu().numpy())

        if (not faces):
            faces = smpl.faces

    # convert vertices to numpy array
    vertices = np.concatenate(vertices, axis=0)
    visualize_frames_open3d(vertices, faces, fps=30)

def visualize_mocap_output(file_loc, smpl, config, framerate=240):
    cdata = np.load(file_loc, allow_pickle=True)
    pose = cdata['fullpose'].astype(np.float32)
    print(pose.shape)

    body_parms = {
        'global_orient': torch.Tensor(pose[:, :3]).to(config.device), # controls the global root orientation
        'body_pose': torch.Tensor(pose[:, 3:66]).to(config.device), # controls the body
    }

    smpl_params = utils.default_smpl_input(pose.shape[0], config)

    for key in body_parms.keys():
        smpl_params[key] = body_parms[key]

    output = smpl(**{k: v for k, v in smpl_params.items()})
    vertices = output.vertices.detach().cpu().numpy()
    faces = smpl.faces

    visualize_frames_open3d(vertices, faces, fps=framerate)

def visualize_frames_open3d(frames, faces, fps=120):
    """
    Visualize a list of (N, 3) point sets in an Open3D window as a mesh,
    refreshing at the specified fps (default 120Hz).

    Args:
        frames (list or array-like): A sequence of arrays, each of shape (N, 3).
                                     Each array in 'frames' represents the vertex
                                     positions for that frame.
        faces (array-like): An array of shape (M, 3), each row containing the vertex
                            indices for one triangular face.
        fps (int): Frames per second to update the visualization.
    """

    # Create a TriangleMesh geometry
    mesh = o3d.geometry.TriangleMesh()

    # Create a Visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D Mesh', width=1280, height=720)

    # Optionally add a coordinate frame (useful for reference)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)

    # Add the initial mesh to the visualizer
    vis.add_geometry(mesh)

    # Calculate the time interval between frames
    frame_interval = 1.0 / fps

    mesh.triangles = o3d.utility.Vector3iVector(faces)

    for i, frame_data in enumerate(frames):
        # Update the mesh's vertices
        mesh.vertices = o3d.utility.Vector3dVector(frame_data)
        mesh.compute_vertex_normals()

        # Update geometry in the visualizer
        vis.update_geometry(mesh)
        vis.poll_events()

        # Optionally reset the viewpoint on the first frame
        if i == 0:
            vis.reset_view_point(True)

        vis.update_renderer()

        # Wait briefly to maintain your desired fps
        time.sleep(frame_interval)

    # Once done, close the window
    vis.destroy_window()

if __name__ == "__main__":
    # process command line arguments
    parser = argparse.ArgumentParser(description="Visualize a SMPL model output.")

    # Optional flag -m
    parser.add_argument('-m', action='store_true', help='the file being visualized is the output of the model in the format (B, config.max_length, 22, 3)')

    # Required positional argument
    parser.add_argument('file_loc', type=str, help='location of file being visualized')

    args = parser.parse_args()

    config = c.config()

    # instantiate a smplx model
    smpl = smplx.create(config.body_model, model_type='smplx',
                            gender='neutral', use_face_contour=False,
                            batch_size=1,
                            ext='npz',
                            age='adult').to(config.device)
    
    if args.m:
        visualize_model_output(args.file_loc, smpl, config)
    else:
        visualize_mocap_output(args.file_loc, smpl, config)
