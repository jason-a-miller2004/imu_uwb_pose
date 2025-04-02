import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def r6d_to_axis_angle(r6d: torch.Tensor) -> np.ndarray:
    """
    Converts a 6D rotation representation (batch_size x 6) to axis-angle (batch_size x 3).
    
    Args:
        r6d: (B, 6) tensor. Each row is [v1, v2] where v1, v2 are in R^3.
             Typically these come from the first two columns of a rotation matrix 
             or something that can be orthonormalized into a rotation.
    
    Returns:
        axis_angles: (B, 3) NumPy array of axis-angle rotations (in radians).
                     Each row is an axis scaled by the rotation magnitude.
    """
    rot_mats = r6d_to_rotation_matrix(r6d)
    
    # 5) Convert rotation matrices to axis-angle (in radians) via SciPy

    # convert to numpy and then to scipy rotation
    rot_obj = R.from_matrix(rot_mats.detach().cpu().numpy())
    # convert to axis-angle
    aa = rot_obj.as_rotvec()  # shape (B,3)
    
    return aa

def r6d_to_rotation_matrix(r6d: torch.Tensor) -> np.ndarray:
    assert r6d.shape[1] == 6, "r6d must have shape (B,6)."
    v1 = r6d[:, 0:3]  # (B,3)
    v2 = r6d[:, 3:6]  # (B,3)
    
    # 1) Normalize v1
    v1_norm = torch.nn.functional.normalize(v1, dim=1)  # (B,3)
    
    # 2) Make v2 orthogonal to v1
    dot = torch.sum(v2 * v1_norm, dim=1, keepdim=True)  # (B,1)
    proj = dot * v1_norm                                # (B,3)
    v2_ortho = v2 - proj                                # (B,3)
    v2_norm = torch.nn.functional.normalize(v2_ortho, dim=1)  # (B,3)
    
    # 3) Compute the 3rd orthonormal vector by cross product
    v3_norm = torch.cross(v1_norm, v2_norm, dim=1)  # (B,3)
    
    # 4) Stack columns to get rotation matrices (B,3,3)
    #    Each slice is a valid rotation matrix if v1, v2 were not degenerate
    rot_mats = torch.stack([v1_norm, v2_norm, v3_norm], dim=2)  # (B,3,3)
    return rot_mats