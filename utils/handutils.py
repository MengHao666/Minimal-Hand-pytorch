import numpy as np
import torch

try:
    from PIL import Image
except ImportError:
    print('Could not import PIL in handutils')
import config as cfg


def get_joint_bone(joint, ref_bone_link=None):
    if ref_bone_link is None:
        ref_bone_link = (0, 9)

    if (
            not torch.is_tensor(joint)
            and not isinstance(joint, np.ndarray)
    ):
        raise TypeError('joint should be ndarray or torch tensor. Got {}'.format(type(joint)))
    if (
            len(joint.shape) != 3
            or joint.shape[1] != 21
            or joint.shape[2] != 3
    ):
        raise TypeError('joint should have shape (B, njoint, 3), Got {}'.format(joint.shape))

    batch_size = joint.shape[0]
    bone = 0
    if torch.is_tensor(joint):
        bone = torch.zeros((batch_size, 1)).to(joint.device)
        for jid, nextjid in zip(
                ref_bone_link[:-1], ref_bone_link[1:]
        ):
            bone += torch.norm(
                joint[:, jid, :] - joint[:, nextjid, :],
                dim=1, keepdim=True
            )  # (B, 1)
    elif isinstance(joint, np.ndarray):
        bone = np.zeros((batch_size, 1))
        for jid, nextjid in zip(
                ref_bone_link[:-1], ref_bone_link[1:]
        ):
            bone += np.linalg.norm(
                (joint[:, jid, :] - joint[:, nextjid, :]),
                ord=2, axis=1, keepdims=True
            )  # (B, 1)
    return bone


def uvd2xyz(
        uvd,
        joint_root,
        joint_bone,
        intr=None,
        trans=None,
        scale=None,
        inp_res=256,
        mode='persp'
):
    bs = uvd.shape[0]
    if mode in ['persp', 'perspective']:
        if intr is None:
            raise Exception("No intr found in perspective")
        '''1. denormalized uvd'''
        uv = uvd[:, :, :2] * inp_res  # 0~256
        depth = (uvd[:, :, 2] * cfg.DEPTH_RANGE) + cfg.DEPTH_MIN
        root_depth = joint_root[:, -1].unsqueeze(-1)  # (B, 1)
        z = depth * joint_bone.expand_as(uvd[:, :, 2]) + \
            root_depth.expand_as(uvd[:, :, 2])  # B x M

        '''2. uvd->xyz'''
        camparam = torch.zeros((bs, 4)).float().to(intr.device)  # (B, 4)
        camparam[:, 0] = intr[:, 0, 0]  # fx
        camparam[:, 1] = intr[:, 1, 1]  # fx
        camparam[:, 2] = intr[:, 0, 2]  # cx
        camparam[:, 3] = intr[:, 1, 2]  # cy
        camparam = camparam.unsqueeze(1).expand(-1, uvd.size(1), -1)  # B x M x 4
        xy = ((uv - camparam[:, :, 2:4]) / camparam[:, :, :2]) * \
             z.unsqueeze(-1).expand_as(uv)  # B x M x 2
        return torch.cat((xy, z.unsqueeze(-1)), -1)  # B x M x 3
    elif mode in ['ortho', 'orthogonal']:
        if trans is None or scale is None:
            raise Exception("No trans or scale found in orthorgnal")
        raise Exception("orth Unimplement !")
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'ortho']")


def xyz2uvd(
        xyz,
        joint_root,
        joint_bone,
        intr=None,
        trans=None,
        scale=None,
        inp_res=256,
        mode='persp'
):
    bs = xyz.shape[0]
    if mode in ['persp', 'perspective']:
        if intr is None:
            raise Exception("No intr found in perspective")
        z = xyz[:, :, 2]
        xy = xyz[:, :, :2]
        xy = xy / z.unsqueeze(-1).expand_as(xy)

        ''' 1. normalize depth : root_relative, scale_invariant '''
        root_depth = joint_root[:, -1].unsqueeze(-1)  # (B, 1)
        depth = (z - root_depth.expand_as(z)) / joint_bone.expand_as(z)

        '''2. xy->uv'''
        camparam = torch.zeros((bs, 4)).float().to(intr.device)  # (B, 4)
        camparam[:, 0] = intr[:, 0, 0]  # fx
        camparam[:, 1] = intr[:, 1, 1]  # fx
        camparam[:, 2] = intr[:, 0, 2]  # cx
        camparam[:, 3] = intr[:, 1, 2]  # cy
        camparam = camparam.unsqueeze(1).expand(-1, xyz.size(1), -1)  # B x M x 4
        uv = (xy * camparam[:, :, :2]) + camparam[:, :, 2:4]

        '''3. normalize uvd to 0~1'''
        uv = uv / inp_res
        depth = (depth - cfg.DEPTH_MIN) / cfg.DEPTH_RANGE

        return torch.cat((uv, depth.unsqueeze(-1)), -1)
    elif mode in ['ortho', 'orthogonal']:
        if trans is None or scale is None:
            raise Exception("No trans or scale found in orthorgnal")
        raise Exception("orth Unimplement !")
    else:
        raise Exception("Unkonwn proj type. should in ['persp', 'ortho']")


def persp_joint2kp(joint, intr):
    joint_homo = torch.matmul(joint, intr.transpose(1, 2))
    kp2d = joint_homo / joint_homo[:, :, 2:]
    kp2d = kp2d[:, :, :2]
    return kp2d


def rot_kp2d(kp2d, rot):
    kp2d = np.concatenate((kp2d, np.ones((kp2d.shape[0], 1))), axis=1)
    new_kp2d = np.matmul(kp2d, rot.transpose())
    return new_kp2d


def get_annot_scale(annots, visibility=None, scale_factor=2.0):
    """
    Retreives the size of the square we want to crop by taking the
    maximum of vertical and horizontal span of the hand and multiplying
    it by the scale_factor to add some padding around the hand
    """
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    s = max_delta * scale_factor
    return s


def get_mask_mini_scale(mask_, side):
    """
    Retreives the size of the square...
    """
    # mask = np.array(mask_.copy())[:, :, 2:].squeeze()
    mask = mask_.copy().squeeze()
    mask_scale = 0
    # print(mask.shape)
    if side == "l":
        id_left = [i for i in range(2, 18)]
        np.putmask(mask, np.logical_and(mask >= id_left[0], mask <= id_left[-1]), 128)
        seg = np.argwhere(mask == 128)
        # print("seg.shape=",seg.shape)
        seg_rmin, seg_cmin = np.min(seg, axis=0)
        seg_rmax, seg_cmax = np.max(seg, axis=0)
        mask_scale = max(seg_rmax - seg_rmin + 1, seg_cmax - seg_cmin + 1)

    elif side == "r":
        id_right = [i for i in range(18, 34)]
        np.putmask(mask, np.logical_and(mask >= id_right[0], mask <= id_right[-1]), 255)

        seg = np.argwhere(mask == 255)
        seg_rmin, seg_cmin = np.min(seg, axis=0)
        seg_rmax, seg_cmax = np.max(seg, axis=0)
        mask_scale = max(seg_rmax - seg_rmin + 1, seg_cmax - seg_cmin + 1)
    elif side == 0:
        rmin, cmin = mask.min(0)
        rmax, cmax = mask.max(0)
        mask_scale = max(rmax - rmin + 1, cmax - cmin + 1)

    if not mask_scale:
        raise ValueError("mask_scale is 0!")

    return mask_scale


def get_kp2d_mini_scale(annots):
    """
    get mini square to include kp2d
    """
    # print("annots=",annots)
    min_x, min_y = annots.min(0)  # opencv convention
    max_x, max_y = annots.max(0)
    # delta_x = int(max_x - min_x)
    # delta_y = int(max_y - min_y)

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    max_delta = max(delta_x, delta_y)

    # return delta_x + 1 if delta_x > delta_y else delta_y + 1
    return max_delta


# def get_ori_crop_scale(mask, side, kp2d, scale_factor=2.0):
#     mask_mini_scale = get_mask_mini_scale(mask, side)
#     kp2d_mini_scale = get_kp2d_mini_scale(kp2d)
#     ori_crop_scale = max(mask_mini_scale, kp2d_mini_scale)
#
#     # if ori_crop_scale % 2 == 0:
#     #     ori_crop_scale += 2
#     # else:
#     #     ori_crop_scale += 3
#
#     return ori_crop_scale * scale_factor

def get_ori_crop_scale(mask, side, kp2d, mask_flag=True,scale_factor=2.0):
    kp2d_mini_scale = get_kp2d_mini_scale(kp2d)

    ori_crop_scale =kp2d_mini_scale

    # if mask.any()!=None:
    if mask_flag:
        # print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        mask_mini_scale = get_mask_mini_scale(mask, side)
        ori_crop_scale = max(mask_mini_scale, kp2d_mini_scale)

    # if ori_crop_scale % 2 == 0:
    #     ori_crop_scale += 2
    # else:
    #     ori_crop_scale += 3

    return ori_crop_scale * scale_factor

def get_annot_center(annots, visibility=None):
    # Get scale
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    return np.asarray([c_x, c_y])


def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows.astype(int)


def transform_img(img, affine_trans, res):
    """
    Args:
    center (tuple): crop center coordinates
    scale (int): size in pixels of the final crop
    res (tuple): final image size
    """
    trans = np.linalg.inv(affine_trans)

    img = img.transform(
        tuple(res), Image.AFFINE, (trans[0, 0], trans[0, 1], trans[0, 2],
                                   trans[1, 0], trans[1, 1], trans[1, 2])
    )
    return img


##### Original from Obman (buggy) #####
# def get_affine_transform(center, scale, res, rot=0):
#     rot_mat = np.zeros((3, 3))
#     sn, cs = np.sin(rot), np.cos(rot)
#     rot_mat[0, :2] = [cs, -sn]
#     rot_mat[1, :2] = [sn, cs]
#     rot_mat[2, 2] = 1
#     # Rotate center to obtain coordinate of center in rotated image
#     origin_rot_center = rot_mat.dot(center.tolist() + [
#         1,
#     ])[:2]
#     # Get center for transform with verts rotated around optical axis
#     # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
#     # For this, rotate the center but around center of image (vs 0,0 in pixel space)
#     t_mat = np.eye(3)
#     t_mat[0, 2] = -res[1] / 2
#     t_mat[1, 2] = -res[0] / 2
#     t_inv = t_mat.copy()
#     t_inv[:2, 2] *= -1
#     transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [
#         1,
#     ])
#     post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
#     total_trans = post_rot_trans.dot(rot_mat)
#     # check_t = get_affine_transform_bak(center, scale, res, rot)
#     # print(total_trans, check_t)
#     affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
#                                                    scale, res)
#     return total_trans.astype(np.float32), affinetrans_post_rot.astype(
#         np.float32)


def get_affine_transform(center, scale, optical_center, out_res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = - optical_center[0]
    t_mat[1, 2] = - optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = (
        t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1])
    )
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(
        transformed_center[:2], scale, out_res
    )
    return (
        total_trans.astype(np.float32),
        affinetrans_post_rot.astype(np.float32),
    )



######################################
def get_affine_transform_test(center, scale, res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [
        1,
    ])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [
        1,
    ])
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
                                                   scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(
        np.float32)

def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet


def get_affine_transform_bak(center, scale, res, rot):
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / scale
    t[1, 1] = float(res[0]) / scale
    t[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    t[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    t[2, 2] = 1
    if rot != 0:
        rot_mat = np.zeros((3, 3))
        sn, cs = np.sin(rot), np.cos(rot)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t))).astype(np.float32)
    return t, t


def gen_cam_param(joint, kp2d, mode='ortho'):
    if mode in ['persp', 'perspective']:
        kp2d = kp2d.reshape(-1)[:, np.newaxis]  # (42, 1)
        joint = joint / joint[:, 2:]
        joint = joint[:, :2]
        jM = np.zeros((42, 2), dtype="float32")
        for i in range(joint.shape[0]):  # 21
            jM[2 * i][0] = joint[i][0]
            jM[2 * i + 1][1] = joint[i][1]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)

        jM = np.concatenate([jM, pad1, pad2], axis=1)  # (42, 4)
        jMT = jM.transpose()  # (4, 42)print
        jMTjM = np.matmul(jMT, jM)  # (4,4)
        jMTb = np.matmul(jMT, kp2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        return cam_param
    elif mode in ['ortho', 'orthogonal']:
        # ortho only when
        assert np.sum(np.abs(joint[0, :])) == 0
        joint = joint[:, :2]  # (21, 2)
        joint = joint.reshape(-1)[:, np.newaxis]
        kp2d = kp2d.reshape(-1)[:, np.newaxis]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)
        jM = np.concatenate([joint, pad1, pad2], axis=1)  # (42, 3)
        jMT = jM.transpose()  # (3, 42)
        jMTjM = np.matmul(jMT, jM)
        jMTb = np.matmul(jMT, kp2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        return cam_param
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'orth']")
