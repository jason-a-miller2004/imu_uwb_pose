import torch

def forward_kinematics_R(R_local, parent):
    r"""
    :math:`R_global = FK(R_local)`

    Forward kinematics that computes the global rotation of each joint from local rotations.

    Notes
    -----
    A joint's *local* rotation is expressed in its parent's frame.

    A joint's *global* rotation is expressed in the base (root's parent) frame.

    R_local[:, i], parent[i] should be the local rotation and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param R_local: Joint local rotation tensor in shape [*] that can reshape to
                    [frames, 3, 3] (rotation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint global rotation, in shape [num_joint, 3, 3].
    """
    R_local = R_local.view(-1, 22, 3, 3)
    R_global = _forward_tree(R_local, parent, torch.bmm)
    return R_global

def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):
    r"""
    Multiply/Add matrices along the tree branches
    """
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
    x_global = torch.stack(x_global, dim=1)
    return x_global

def get_parent_array(smpl_skeleton, config):
    """
    Given:
        - smpl_skeleton: Tensor of shape (N, 2) listing parent->child edges.
    
    Returns:
        - parent: Tensor of shape (N), where parent[i, j] is the
          parent of joint j in frame i (or -1 if it has no parent).
    """
    num_joints = (torch.max(smpl_skeleton) + 1)

    # Initialize parent array with -1 (indicating root joints)
    parent = torch.full((num_joints,), -1, dtype=torch.long)
    
    # Assign parent indices
    for parent_joint, child_joint in smpl_skeleton:
        parent[child_joint] = parent_joint
    
    return parent

def default_smpl_input(batch_size, config):
    '''
    SMPL batching is setup strange and need to be instantiated on model creation instead of on forward pass unless all member variables are passed in.
    This function creates a default input for SMPL model that can be modified as needed.
    Makes a couple assumptions about the model namely 
        num_expression_coefficients is set to 10 (the default)
        num_betas is set to 10 (the default)
        use_pca is set to true and num_pca_comps is set to 6 (the default)
    '''
    return {
        'global_orient': torch.eye(3, device=config.device).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous(),
        'body_pose': torch.eye(3, device=config.device).view(
                1, 1, 3, 3).expand(
                    batch_size, 21, -1, -1).contiguous(),
        'betas': torch.zeros((batch_size, 10)).to(config.device),
        'transl': torch.zeros((batch_size, 3)).to(config.device),
        'jaw_pose': torch.zeros((batch_size, 3)).to(config.device),
        'left_hand_pose': torch.zeros((batch_size, 6)).to(config.device),
        'right_hand_pose': torch.zeros((batch_size, 6)).to(config.device),
        'expression': torch.zeros((batch_size, 10)).to(config.device),
        'leye_pose': torch.zeros((batch_size, 3)).to(config.device),
        'reye_pose': torch.zeros((batch_size, 3)).to(config.device),
    }

def r6d_to_axis_angle(r6d: torch.Tensor) -> torch.Tensor:
    """
    Converts a 6D rotation representation (batch_size x 6) to axis-angle (batch_size x 3).
    
    Args:
        r6d: (B, 6) tensor. Each row is [v1, v2] where v1, v2 are in R^3.
             Typically these come from the first two columns of a rotation matrix 
             or something that can be orthonormalized into a rotation.
    
    Returns:
        axis_angles: (B, 3) tensor of axis-angle rotations (in radians).
    """
    rot_mats = r6d_to_rotation_matrix(r6d)
    return rotation_matrix_to_axis_angle(rot_mats)

def r6d_to_rotation_matrix(r6d: torch.Tensor) -> torch.Tensor:
    assert r6d.shape[1] == 6, "r6d must have shape (B,6)."

    v1 = r6d[:, 0:3]  # (B,3)
    v2 = r6d[:, 3:6]  # (B,3)
    
    v1_norm = torch.nn.functional.normalize(v1, dim=1)  # (B,3)
    
    dot = torch.sum(v2 * v1_norm, dim=1, keepdim=True)  # (B,1)
    proj = dot * v1_norm                                # (B,3)
    v2_ortho = v2 - proj                                # (B,3)
    v2_norm = torch.nn.functional.normalize(v2_ortho, dim=1)  # (B,3)
    
    v3_norm = torch.cross(v1_norm, v2_norm, dim=1)  # (B,3)
    
    rot_mats = torch.stack([v1_norm, v2_norm, v3_norm], dim=2)  # (B,3,3)

    det = torch.det(rot_mats)
    if torch.any(det < 1e-6):
        rot_mats[det < 1e-6] = torch.eye(3, device=rot_mats.device, dtype=rot_mats.dtype)
    return rot_mats

def rotation_matrix_to_axis_angle(rot_mats: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices (B,3,3) to axis-angle vectors (B,3) using a torch-only log map.
    """
    if rot_mats.ndim < 3:
        raise ValueError("rot_mats must have shape (...,3,3)")
    orig_shape = rot_mats.shape[:-2]
    mats = rot_mats.reshape(-1, 3, 3)
    trace = mats[:, 0, 0] + mats[:, 1, 1] + mats[:, 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    axis = torch.stack([
        mats[:, 2, 1] - mats[:, 1, 2],
        mats[:, 0, 2] - mats[:, 2, 0],
        mats[:, 1, 0] - mats[:, 0, 1],
    ], dim=1)
    sin_theta = torch.sin(theta)
    scale = torch.empty_like(theta)
    mask = sin_theta.abs() > 1e-6
    scale[mask] = theta[mask] / (2.0 * sin_theta[mask])
    scale[~mask] = 0.5 + (trace[~mask] - 3.0) / 12.0
    axis_angle = axis * scale.unsqueeze(1)
    return axis_angle.reshape(*orig_shape, 3)

def rotation_matrix_to_r6d(rot_mats: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices (batch_size x 3 x 3) to 6D representation (batch_size x 6).
    """
    if rot_mats.ndim < 3 or rot_mats.shape[-2:] != (3, 3):
        raise ValueError("rot_mats must have shape (...,3,3).")
    
    mats = rot_mats.reshape(-1, 3, 3)
    v1 = mats[:, :, 0]  # (B,3)
    v2 = mats[:, :, 1]  # (B,3)
    r6d = torch.cat([v1, v2], dim=1)  # (B,6)
    
    new_shape = rot_mats.shape[:-2] + (6,)
    return r6d.reshape(new_shape)

def axis_angle_to_rotation_matrix(axis_angles: torch.Tensor) -> torch.Tensor:
    """
    Converts axis-angle (batch_size x 3) to rotation matrices (batch_size x 3 x 3).
    """
    if axis_angles.ndim < 1 or axis_angles.shape[-1] != 3:
        raise ValueError("axis_angles must have shape (...,3).")
    orig_shape = axis_angles.shape[:-1]
    vecs = axis_angles.reshape(-1, 3)
    angles = torch.linalg.norm(vecs, dim=1, keepdim=True)
    eps = 1e-6
    axis = torch.zeros_like(vecs)
    mask = (angles > eps).squeeze(1)
    axis[mask] = vecs[mask] / angles[mask]
    K = _skew(axis)
    batch = vecs.shape[0]
    eye = torch.eye(3, device=vecs.device, dtype=vecs.dtype).expand(batch, -1, -1)
    angles_flat = angles.squeeze(1)
    sin = torch.sin(angles_flat).unsqueeze(-1).unsqueeze(-1)
    cos = torch.cos(angles_flat).unsqueeze(-1).unsqueeze(-1)
    K2 = K @ K
    rot = eye + sin * K + (1 - cos) * K2
    small = angles_flat < eps
    if torch.any(small):
        rot[small] = eye[small] + K[small] + 0.5 * K2[small]
    return rot.reshape(*orig_shape, 3, 3)

def axis_angle_to_r6d(axis_angles: torch.Tensor) -> torch.Tensor:
    """
    Converts axis-angle (batch_size x 3) to 6D representation (batch_size x 6).
    """
    rot_mats = axis_angle_to_rotation_matrix(axis_angles)
    return rotation_matrix_to_r6d(rot_mats)

def _skew(vecs: torch.Tensor) -> torch.Tensor:
    """
    Build skew-symmetric matrices for a batch of 3D vectors.
    """
    zeros = torch.zeros(vecs.shape[0], device=vecs.device, dtype=vecs.dtype)
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    return torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=1).reshape(-1, 3, 3)

@torch.no_grad()                # 1. turn off autograd
def get_smpl_output(model, pose, config, trans=None):
    """
    Run SMPL in small chunks so GPU VRAM never spikes.
    Returns CPU tensors: vertices  [N, 6890, 3],
                          joints   [N, 144, 3] (SMPL-X) or [N, 24, 3] (SMPL)
    """
    model.eval()                # 2. inference mode

    # If you want FP16, uncomment the next two lines:
    # model.half()              # 4a. halve model weights
    # pose = pose.to(torch.float16)

    chunk = config.batch_size   # 5. single mini-batch at a time
    verts_list, joints_list = [], []

    for start in range(0, pose.shape[0], chunk):
        end = start + chunk
        cur_pose = pose[start:end].to(config.device, non_blocking=True)

        smpl_params = default_smpl_input(cur_pose.shape[0], config)
        smpl_params['global_orient'] = cur_pose[:, :3]
        smpl_params['body_pose']     = cur_pose[:, 3:66]

        if trans is not None:
            cur_trans = trans[start:end].to(config.device, non_blocking=True)
            smpl_params['transl'] = cur_trans

        output = model(**smpl_params)

        # 3. copy results to CPU immediately, then free GPU tensors
        verts_list.append(output.vertices.cpu())
        joints_list.append(output.joints.cpu())

        del output, cur_pose, smpl_params
        torch.cuda.empty_cache()    # releases cached blocks

    vertices = torch.cat(verts_list, dim=0)
    joints   = torch.cat(joints_list, dim=0)
    return vertices, joints, model.faces
