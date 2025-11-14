# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import math
import torch

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

def make_color_wheel_torch(device='cpu'):
    """创建颜色轮 (PyTorch版本)"""
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros(ncols, 3, device=device)
    
    col = 0
    # RY区域
    colorwheel[0:RY, 0] = 1
    colorwheel[0:RY, 1] = torch.arange(0, 1, 1/RY, device=device)
    col += RY
    
    # YG区域
    colorwheel[col:col+YG, 0] = torch.arange(1, 0, -1/YG, device=device)
    colorwheel[col:col+YG, 1] = 1
    col += YG
    
    # GC区域
    colorwheel[col:col+GC, 1] = 1
    colorwheel[col:col+GC, 2] = torch.arange(0, 1, 1/GC, device=device)
    col += GC
    
    # CB区域
    colorwheel[col:col+CB, 1] = torch.arange(1, 0, -1/CB, device=device)
    colorwheel[col:col+CB, 2] = 1
    col += CB
    
    # BM区域
    colorwheel[col:col+BM, 2] = 1
    colorwheel[col:col+BM, 0] = torch.arange(0, 1, 1/BM, device=device)
    col += BM
    
    # MR区域
    colorwheel[col:col+MR, 2] = torch.arange(1, 0, -1/MR, device=device)
    colorwheel[col:col+MR, 0] = 1
    
    return colorwheel

def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

def compute_color_torch(u, v):
    """
    PyTorch版本的光流颜色计算
    :param u: 水平光流图 [H, W]
    :param v: 垂直光流图 [H, W]
    :return: RGB颜色编码 [H, W, 3]
    """
    device = u.device
    b, h, w = u.shape
    img = torch.zeros((b, h, w, 3), device=device, dtype=torch.uint8)
    
    # NaN处理
    nan_mask = torch.isnan(u) | torch.isnan(v)
    u = torch.where(nan_mask, torch.tensor(0.0, device=device), u)
    v = torch.where(nan_mask, torch.tensor(0.0, device=device), v)
    
    # 获取颜色轮
    colorwheel = make_color_wheel_torch(device)
    ncols = colorwheel.size(0)
    
    # 计算幅度和角度
    rad = torch.sqrt(u**2 + v**2)
    a = torch.atan2(-v, -u) / math.pi
    
    # 颜色索引计算
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1 = torch.where(k1 == ncols + 1, torch.tensor(1, device=device), k1)
    f = fk - k0.float()
    
    # 颜色插值
    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1]  # 已经是0-255范围
        col1 = tmp[k1 - 1]
        col = (1 - f) * col0 + f * col1
        
        # 幅度调整
        low_rad_mask = rad <= 1
        col = torch.where(low_rad_mask, 1 - rad * (1 - col), col * 0.75)
        
        # NaN区域置零
        col = torch.where(nan_mask, torch.tensor(0.0, device=device), col)
        
        # 确保在0-255范围内并转为uint8
        img[:, :, :, i] = (col * 255).clamp(0, 255).to(torch.uint8)
    
    return img


def flow_to_image_torch(flow):
    """
    PyTorch版本的光流转RGB图像
    :param flow: 光流张量 [H, W, 2]
    :return: RGB图像 [H, W, 3] (uint8)
    """
    device = flow.device
    u = flow[..., 0]
    v = flow[..., 1]

    # 处理未知流
    unknown_mask = (torch.abs(u) > UNKNOWN_FLOW_THRESH) | (torch.abs(v) > UNKNOWN_FLOW_THRESH)
    u = torch.where(unknown_mask, torch.tensor(0.0, device=device), u)
    v = torch.where(unknown_mask, torch.tensor(0.0, device=device), v)
    
    # 计算最大幅度
    rad = torch.sqrt(u**2 + v**2)
    maxrad = torch.max(rad)
    
    # 归一化
    eps = torch.finfo(rad.dtype).eps
    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    
    # 计算颜色
    img = compute_color_torch(u, v)
    
    # 未知流区域置零
    unknown_mask_3d = unknown_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
    img = torch.where(unknown_mask_3d, torch.tensor(0, device=device), img)
    
    return img

def save_flow_torch(flow, output_path):
    """保存PyTorch光流可视化结果"""
    from PIL import Image
    flow_img = flow_to_image_torch(flow)
    # 转换到CPU并转为numpy数组
    flow_img_np = flow_img.cpu().numpy()
    for i in range(flow_img.shape[0]):
        print(f"Saving flow image {i:04d} to {output_path}/{i:04d}_flow.png")
        Image.fromarray(flow_img_np[i]).save(f"{output_path}/{i:04d}_flow.png")

def flow_tensor_to_image_torch(flow):
    """将光流张量转换为图像格式 [3, H, W]"""
    # 确保输入是[H, W, 2]
    if flow.dim() == 3 and flow.size(0) == 2:
        flow = flow.permute(1, 2, 0)
    
    # 转换为RGB图像
    flow_img = flow_to_image_torch(flow)
    
    # 转换为[3, H, W]格式
    return flow_img.permute(2, 0, 1)

def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def save_vis_flow_tofile(flow, output_path):
    vis_flow = flow_to_image(flow)
    Image.fromarray(vis_flow).save(output_path)


def flow_tensor_to_image(flow):
    """Used for tensorboard visualization"""
    flow = flow.permute(1, 2, 0)  # [H, W, 2]
    flow = flow.detach().cpu().numpy()
    flow = flow_to_image(flow)  # [H, W, 3]
    flow = np.transpose(flow, (2, 0, 1))  # [3, H, W]

    return flow
