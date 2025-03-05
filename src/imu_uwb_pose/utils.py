import torch
import numpy as np

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