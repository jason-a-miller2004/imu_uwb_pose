import torch
import numpy as np
import smplx
from imu_uwb_pose import config as c, utils, data_extraction as de
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import argparse
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R

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

def visualize_mocap_output(file_loc, smpl, config, framerate=30):
    cdata = np.load(file_loc, allow_pickle=True)
    pose = cdata['fullpose'].astype(np.float32)

    # currently at 120hz resample to 30hz
    pose = torch.tensor(pose[::4, :])
    
    vertices, joints, faces = utils.get_smpl_output(smpl, pose, config)

    visualize_frames_open3d(vertices, faces, fps=framerate)

def align_output(sensor_loc, mocap_loc, config, overlay, offset):
    cdata = np.load(mocap_loc, allow_pickle=True)
    pose = cdata['fullpose'].astype(np.float32)

    # currently at 120hz resample to 30hz
    pose = torch.tensor(pose[::4, :])
    vertices, joints, faces = utils.get_smpl_output(smpl, pose, config)

    with open(sensor_loc, 'rb') as file:
        data = pickle.load(file)

    # plot uwb distances and imu angles
    uwb_dists = data['uwb'][offset[0]:offset[1]]
    smpl_dists = de.extract_uwb_amass(joints, config)

    # make a subplot with two plots showing both dists with number of frames being the x-axis
    plot_and_compare(smpl_dists, uwb_dists, overlay, title='UWB Distances', smpl_ylabel='SMPL distance', sensor_ylabel='Sensor distance')

    # convert left and right imus to axis angle
    left_imu_ori = data['left_imu']
    right_imu_ori = data['right_imu']
    left_imu_ori = R.from_matrix(left_imu_ori)
    left_imu_ori = left_imu_ori.as_rotvec()[offset[0]:offset[1]]

    right_imu_ori = R.from_matrix(right_imu_ori)
    right_imu_ori = right_imu_ori.as_rotvec()[offset[0]:offset[1]]

    # get left and right from smpl
    joint_angles = de.extract_angle_amass(pose, config)
    left_smpl_ori = joint_angles[:, 0, :]
    right_smpl_ori = joint_angles[:, 1, :]

    # plot and compare left ankle
    plot_and_compare(left_smpl_ori, left_imu_ori, overlay, 'Left Ankle Orientation', smpl_ylabel='SMPL Orientation', sensor_ylabel='Sensor Orientation', labels=(['sensor_x', 'sensor_y', 'sensor_z'], ['smpl_x', 'smpl_y', 'smpl_z']))

    # plot and compare right ankle
    plot_and_compare(right_smpl_ori, right_imu_ori, overlay, 'Right ankle Orientation', smpl_ylabel='SMPL Orientation', sensor_ylabel='Sensor Orientation', labels=(['sensor_x', 'sensor_y', 'sensor_z'], ['smpl_x', 'smpl_y', 'smpl_z']))

def plot_and_compare(smpl_output, sensor_output, overlay, title, smpl_ylabel, sensor_ylabel, labels=None):
    # plot the smpl output and sensor output
    if labels is None:
            labels = ('Sensor Output', 'SMPL Output')

    if (overlay):
    
        plt.plot(sensor_output, label=labels[0])
        plt.plot(smpl_output, label=labels[1])
        plt.title(title)
        plt.xlabel('Frame Index')
        plt.ylabel(smpl_ylabel + ' and '+ sensor_ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # UWB distances on the top
        ax1.plot(sensor_output, label=labels[0])
        ax1.set_ylabel(sensor_ylabel)
        ax1.set_title(title)
        ax1.legend()

        # SMPL distances below
        ax2.plot(smpl_output, label=labels[1])
        ax2.set_ylabel(smpl_ylabel)
        ax2.set_xlabel('Frame Index')
        ax2.legend()

        plt.tight_layout()
        plt.show()

def visualize_frames_open3d(frames, faces, fps=30):
    gui.Application.instance.initialize()
    w = gui.Application.instance.create_window("Open3D mesh", 1280, 720)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    w.add_child(scene)

    # --- geometry ----------------------------------------------------------------
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertices = o3d.utility.Vector3dVector(frames[0])
    mesh.compute_vertex_normals()
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    scene.scene.add_geometry("mesh", mesh, mat)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    scene.scene.add_geometry("axis", axis, rendering.MaterialRecord())

    bbox = mesh.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bbox, bbox.get_center())

    # --- overlay label ------------------------------------------------------------
    counter = gui.Label("")
    counter.text_color = gui.Color(1, 1, 1)                    # white
    counter.background_color = gui.Color(0, 0, 0, 0.5)         # 50 % black
    w.add_child(counter)

    def on_layout(ctx):                       # <- ctx is the LayoutContext object
        # The second argument is usually an *empty* constraints object.
        pref = counter.calc_preferred_size(ctx, gui.Widget.Constraints())

        r = w.content_rect                     # window’s drawable rectangle
        scene.frame = r              # 🔹 GIVE THE SceneWidget A FRAME 🔹

        # Place the label 8 px from the top-right corner
        counter.frame = gui.Rect(r.get_right() - (pref.width + 32) - 8,
                                r.y + 8,
                                pref.width + 32,
                                pref.height)
    w.set_on_layout(on_layout)

    # --- animation state ----------------------------------------------------------
    total = len(frames)
    idx   = 0
    step  = 1.0 / fps
    last  = [time.time()]

    def on_tick():                                             # <- replaces call_later
        nonlocal idx
        now = time.time()
        if now - last[0] < step:                               # throttle to target FPS
            return False                                       # no redraw needed
        last[0] = now

        mesh.vertices = o3d.utility.Vector3dVector(frames[idx])
        mesh.compute_vertex_normals()
        scene.scene.remove_geometry("mesh")
        scene.scene.add_geometry("mesh", mesh, mat)
        counter.text = f"{idx + 1} / {total}"

        idx = (idx + 1) % total
        w.post_redraw()                                        # ask window to repaint
        return True                                            # we did change things

    w.set_on_tick_event(on_tick)                               # <-- this is the key line
    gui.Application.instance.run()

if __name__ == "__main__":
    # process command line arguments
    parser = argparse.ArgumentParser(description="Visualize a SMPL model output.")

    # Optional flag -m
    parser.add_argument('-m', action='store_true', help='the file being visualized is the output of the model in the format (B, config.max_length, 22, 3)')

    parser.add_argument('-o', action='store_true', help='if aligning overlay the graphs on top of each other')

    parser.add_argument(
        "--align",
        type=str,
        help="Path to the sensor file"
    )

    # Required positional argument
    parser.add_argument('file_loc', type=str, help='location of file being visualized')

    parser.add_argument("--offset", nargs=2, type=int, metavar=("start", "stop"),
                        help="Provide exactly two integers")

    # offset to 
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
    elif args.align is not None:
        align_output(args.align, args.file_loc, config, args.o, args.offset)
    else:
        visualize_mocap_output(args.file_loc, smpl, config)
