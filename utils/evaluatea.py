
import os
import numpy as np
import torch
import torch.nn.functional as F

from model_pcv.Hyperparameters import TILE_IN_F,F_IN_GOF,QUALITY_LEVELS,VIDEO_GOF_LEN,\
    FRAME,INIT_QOE,REBUF_PENALTY,SMOOTH_PENALTY,MULTIPLE_QUALITY_LEVELS
from utils.logger import setup_logger


def evaluate(self, fov_traces, dis_traces, num_traces=5):
    """
    评估模型性能，不使用真实FOV
    """
    # 保存当前训练状态
    training_policy = self.policy.modules_except_plm.training
    training_plm = self.plm.training if self.args.rank != -1 else False
    
    # 设置为评估模式
    self.policy.modules_except_plm.eval()
    if self.args.rank != -1:
        self.plm.eval()
    
    # 评估统计
    all_rewards = []
    all_qualities = []
    all_rebuffers = []
    all_switches = []
    
    # 限制评估轨迹数量
    eval_trace_indices = np.random.choice(len(fov_traces), min(num_traces, len(fov_traces)), replace=False)
    
    with torch.no_grad():
        for trace_idx in eval_trace_indices:
            user_fov_trace = fov_traces[trace_idx]
            user_dis_trace = dis_traces[trace_idx]
            
            # 重置环境和历史
            self.env.reset()
            self.fov_history = []
            self.policy.clear_dq()
            
            # 轨迹统计
            trace_reward = []
            trace_qualities = []
            trace_rebuffer = []
            trace_switch = []
            
            # 模拟视频播放
            current_frame = 0
            
            while current_frame < len(self.video_size):
                delay, rebuffer, quality, switch, selected_tile, selected_quality, buffer = self.stream_next_gof(
                    user_fov_trace, 
                    user_dis_trace,
                    current_frame
                )
                
                # 计算奖励和惩罚
                quality_reward = quality * INIT_QOE
                rebuffer_penalty = rebuffer * REBUF_PENALTY
                switch_penalty = switch * SMOOTH_PENALTY
                reward = quality_reward - rebuffer_penalty - switch_penalty
                
                # 更新统计
                trace_qualities.append(quality)
                trace_rebuffer.append(rebuffer)
                trace_switch.append(switch)
                trace_reward.append(reward)
                
                # 更新当前帧
                current_frame += F_IN_GOF
                
                if current_frame >= len(self.video_size):
                    break
            
            # 记录轨迹结果
            all_rewards.append(np.sum(trace_reward))
            all_qualities.append(np.sum(trace_qualities) if trace_qualities else 0)
            all_rebuffers.append(np.sum(trace_rebuffer))
            all_switches.append(np.sum(trace_switch))
    
    # 恢复训练状态
    if training_policy:
        self.policy.modules_except_plm.train()
    if self.args.rank != -1 and training_plm:
        self.plm.train()
        
    # 清理内存
    torch.cuda.empty_cache()
    
    # 计算平均值
    avg_reward = np.mean(all_rewards)
    avg_quality = np.mean(all_qualities)
    avg_rebuffer = np.mean(all_rebuffers)
    avg_switch = np.mean(all_switches)
    
    return avg_reward, avg_quality, avg_rebuffer, avg_switch
