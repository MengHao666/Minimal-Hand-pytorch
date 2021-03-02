import torch
import cv2
import numpy as np
import random
import torchvision
import utils.func as func
import config as cfg


def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue)

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img


def batch_with_dep(clrs, deps):
    clrs = func.to_numpy(clrs)
    if clrs.dtype is not np.uint8:
        clrs = (clrs * 255).astype(np.uint8)
    assert len(deps.shape) == 4, "deps should have shape (B, 1, H, W)"
    deps = func.to_numpy(deps)
    deps = deps.swapaxes(1, 2).swapaxes(2, 3)
    deps = deps.repeat(3, axis=3)
    if deps.dtype is not np.uint8:
        deps = (deps * 255).astype(np.uint8)

    batch_size = clrs.shape[0]

    alpha = 0.6
    beta = 0.9
    gamma = 0

    batch = []
    for i in range(16):
        if i >= batch_size:
            batch.append(np.zeros((64, 64, 3)).astype(np.uint8))
            continue
        clr = clrs[i]
        clr = cv2.resize(clr, (64, 64))
        dep = deps[i]
        dep_img = cv2.addWeighted(clr, alpha, dep, beta, gamma)
        batch.append(dep_img)

    resu = []
    for i in range(4):
        resu.append(np.concatenate(batch[i * 4: i * 4 + 4], axis=1))
    resu = np.concatenate(resu)
    return resu


def batch_with_joint(clrs, uvds):
    clrs = func.to_numpy(clrs)
    if clrs.dtype is not np.uint8:
        clrs = (clrs * 255).astype(np.uint8)
    uvds = func.to_numpy(uvds)

    batch_size = clrs.shape[0]

    batch = []
    for i in range(16):
        if i >= batch_size:
            batch.append(np.zeros((256, 256, 3)).astype(np.uint8))
            continue
        clr = clrs[i]
        uv = (np.array(uvds[i][:, :2]) * clr.shape[0]).astype(np.uint8)  # (256)
        clr = draw_hand_skeloten(clr, uv, cfg.SNAP_BONES)
        batch.append(clr)

    resu = []
    for i in range(4):
        resu.append(np.concatenate(batch[i * 4: i * 4 + 4], axis=1))
    resu = np.concatenate(resu)
    return resu


def draw_hand_skeloten(clr, uv, bone_links, colors=cfg.JOINT_COLORS):
    for i in range(len(bone_links)):
        bone = bone_links[i]
        for j in bone:
            cv2.circle(clr, tuple(uv[j]), 4, colors[i], -1)
        for j, nj in zip(bone[:-1], bone[1:]):
            cv2.line(clr, tuple(uv[j]), tuple(uv[nj]), colors[i], 2)
    return clr


def batch_with_heatmap(
        inputs,
        heatmaps,
        num_rows=2,
        parts_to_show=None,
        n_in_batch=1,
):
    # inputs = func.to_numpy(inputs * 255)  # 0~1 -> 0 ~255
    heatmaps = func.to_numpy(heatmaps)
    batch_img = []
    for n in range(min(inputs.shape[0], n_in_batch)):
        inp = inputs[n]
        batch_img.append(
            sample_with_heatmap(
                inp,
                heatmaps[n],
                num_rows=num_rows,
                parts_to_show=parts_to_show
            )
        )
    resu = np.concatenate(batch_img)
    return resu


def sample_with_heatmap(img, heatmap, num_rows=2, parts_to_show=None):
    if parts_to_show is None:
        parts_to_show = np.arange(heatmap.shape[0])  # 21

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = cv2.resize(img, (size, size))

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = cv2.resize(heatmap[part_idx], (size, size))
        out_resized = out_resized.astype(float)
        out_img = inp_small.copy() * .4
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .6

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def color_heatmap(x):
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:, :, 1] = gauss(x, 1, .5, .3)
    color[:, :, 2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d
