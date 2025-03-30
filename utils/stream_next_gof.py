
import os
import numpy as np
import torch
import torch.nn.functional as F

from model_pcv.Hyperparameters import TILE_IN_F,F_IN_GOF,QUALITY_LEVELS,VIDEO_GOF_LEN,\
    FRAME,INIT_QOE,REBUF_PENALTY,SMOOTH_PENALTY,MULTIPLE_QUALITY_LEVELS
from utils.logger import setup_logger

def stream_next_gof(self, user_fov_trace, user_dis_trace, current_frame):
    """
    流式传输下一个GOF，完全基于模型预测
    
    参数:
        user_fov_trace: 用户FOV轨迹 (仅用于计算奖励，不用于决策)
        user_dis_trace: 用户距离轨迹 (仅用于计算奖励，不用于决策)
        current_frame: 当前帧索引
            
    返回:
        延迟、重缓冲时间、质量等指标
    """
    # 更新FOV历史
    self.playback_pos = int(self.env.playback_position)
    self.fov_history = []
    self.dis_history = []
    for i in range(max(0, self.playback_pos - self.history_length), self.playback_pos):
        if i < len(user_fov_trace):
            self.fov_history.append(user_fov_trace[i])
            self.dis_history.append(user_dis_trace[i])
    
    # 准备环境状态
    state = self.preprocess_state()
    
    # 使用策略网络直接预测质量级别
    with torch.no_grad():
        # 计算目标回报
        if len(self.stats['episode_rewards']) > 0:
            best_reward = max(self.stats['episode_rewards'])
            recent_avg = np.mean(self.stats['episode_rewards'][-10:])
            target_return = max(best_reward, recent_avg) * 1.1
        else:
            # 初始估计
            quality_sum = 0
            for i in range(F_IN_GOF):
                for j in range(TILE_IN_F):
                    quality_sum += self.video_size[0+i][j][3]
            target_return = INIT_QOE * quality_sum
        
        # 当前时间步
        timestep = self.env.video_frame_counter // F_IN_GOF
        
        # 获取质量级别决策
        selected_tile,quality_levels = self.policy.sample(
            state=state,
            target_return=target_return,
            timestep=timestep
        )
    self.logger.info(f"tile:{selected_tile},quality:{quality_levels}")
    # 执行下载
    delay, sleep_time, buffer_size, rebuffer, gof_size, done, gof_remain, buffer = \
        self.env.get_video_gof(selected_tile, quality_levels)
    
    # 仅用于评估 - 计算实际FOV中的tiles
    seen = [0] * TILE_IN_F
    for i in range(TILE_IN_F):
        for f in range(current_frame, min(current_frame + F_IN_GOF, len(user_fov_trace))):
            if f < len(user_fov_trace) and user_fov_trace[f][i]:
                seen[i] = 1
                break
    
    # 计算平均质量
    quality = 0
    for i in range(F_IN_GOF):
        frame_idx = min(current_frame + i, len(self.video_size) - 1)
        for j in range(TILE_IN_F):
            if seen[j] and selected_tile[j]:
                dis_idx = min(frame_idx, len(user_dis_trace) - 1)
                quality += self.video_size[frame_idx][j][quality_levels[j]] / user_dis_trace[dis_idx][j]
    
    # 计算切换量
    switch = 0.0
    if current_frame > 0:
        for s in range(TILE_IN_F):
            if selected_tile[s]:
                prev_quality = max(0, buffer[current_frame//F_IN_GOF-1][s])
                switch += np.abs(
                    MULTIPLE_QUALITY_LEVELS[quality_levels[s]] - 
                    MULTIPLE_QUALITY_LEVELS[prev_quality]
                )
    reward=quality*INIT_QOE-rebuffer*REBUF_PENALTY-switch*SMOOTH_PENALTY
    # if self.policy.training:
    #     self.logger.info(f"预测的质量级别: {quality_levels}, 选择的tile: {selected_tile}, 奖励: {reward:.4f}")
    
    return delay, rebuffer, quality, switch, selected_tile, quality_levels, buffer