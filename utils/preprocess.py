
import os
import numpy as np
import torch
import torch.nn.functional as F

from model_pcv.Hyperparameters import TILE_IN_F,F_IN_GOF,QUALITY_LEVELS,VIDEO_GOF_LEN,\
    FRAME,INIT_QOE,REBUF_PENALTY,SMOOTH_PENALTY,MULTIPLE_QUALITY_LEVELS
from utils.logger import setup_logger


def preprocess_state(self):
    """预处理环境状态为张量格式，只使用历史数据"""
    # 1. 带宽特征
    bandwidth_features = []
    ptr = self.env.mahimahi_ptr
    for i in range(3):  # 过去3个带宽样本
        if ptr > 0:
            ptr -= 1
            bandwidth_features.append(self.env.cooked_bw[ptr] / max(self.env.cooked_bw[ptr:ptr+10]))
        else:
            bandwidth_features.append(0)
    
    # 2. FOV历史
    fov_features = []
    # 如果历史不足50帧，用1填充
    if len(self.fov_history) < self.history_length:
        padding_length = self.history_length - len(self.fov_history)
        fov_features.extend([1] * (padding_length * TILE_IN_F))
        # 添加已有的历史
        for fov in self.fov_history:
            fov_features.extend(fov)
    else:
        # 使用最近的50帧FOV
        for fov in self.fov_history[-self.history_length:]:
            fov_features.extend(fov)
    
    # 3. 缓冲区特征
    buffer_features = []
    # 缓冲区大小
    buffer_features.append(self.env.buffer_size)  # 秒
    
    # 当前GOF的缓冲区状态
    current_gof_index = self.env.video_frame_counter // F_IN_GOF
    if current_gof_index < len(self.env.buffer):
        gof_buffer = self.env.buffer[current_gof_index]  # 获取当前GOF的缓冲区
        # 归一化质量级别
        for quality in gof_buffer:
            if quality == -1:  # 未下载 
                buffer_features.append(0.0)
            else:
                buffer_features.append((quality + 1) / QUALITY_LEVELS)
    else:
        buffer_features.extend([0.0] * TILE_IN_F)
    
    # 4. 下一个GOF的大小信息
    gof_size_features = []
    next_frame = self.env.video_frame_counter
    if next_frame < len(self.video_size) - F_IN_GOF:
        for tile in range(TILE_IN_F):
            for quality in range(QUALITY_LEVELS):
                # 计算该tile该质量级别的GOF大小
                tile_size = 0
                for frame in range(F_IN_GOF):
                    if next_frame + frame < len(self.video_size):
                        tile_size += self.video_size[next_frame + frame][tile][quality]
                
                # 归一化大小
                gof_size_features.append(tile_size)  # 为Mb
    else:
        # 已经是最后一个GOF，填充0
        gof_size_features.extend([0.0] * (TILE_IN_F * QUALITY_LEVELS))
    
    # 5. 视频帧计数器
    frame_counter = [self.env.video_frame_counter / len(self.video_size)]  # 归一化
    
    # 6. 历史距离信息 (与FOV历史同步)
    dis_features = []
    # 如果历史不足50帧，用1填充
    if len(self.dis_history) < self.history_length:
        padding_length = self.history_length - len(self.dis_history)
        dis_features.extend([1.0] * (padding_length * TILE_IN_F))
        # 添加已有的历史
        for dis in self.dis_history:
            dis_features.extend(dis)
    else:
        # 使用最近的50帧FOV
        for dis in self.dis_history[-self.history_length:]:
            dis_features.extend(dis)
    
    # 组合所有特征
    state = np.concatenate([
        bandwidth_features,
        fov_features,
        buffer_features,
        gof_size_features,
        frame_counter,
        dis_features
    ])
    
    return torch.FloatTensor(state).unsqueeze(0)
