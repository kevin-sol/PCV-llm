import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils.logger import setup_logger
from model_pcv.Hyperparameters import F_IN_GOF,VIDEO_GOF_LEN

class PointCloudRLPolicy(nn.Module):
    """
    简化版点云策略网络。
    直接从状态预测每个tile的质量级别，不使用当前FOV信息。
    决策将仅基于历史数据和状态特征。
    """
    def __init__(
            self,
            state_feature_dim,      # 状态特征维度
            tile_num,               # tile数量
            quality_levels,         # 质量级别数
            state_encoder,          # 状态编码器
            plm,                    # 预训练语言模型
            plm_embed_size,         # PLM嵌入维度
            max_length=50,          # 最大序列长度
            max_ep_len=10,          # 最大回合长度
            device='cuda' if torch.cuda.is_available() else 'cpu',  # 运行设备
            device_out = None,      # 输出设备
            residual = False,       # 是否使用残差连接
            which_layer = -1,       # 提前停止层数
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device
        self.logger = setup_logger("policy_log")
        self.tile_num = tile_num 
        self.quality_levels = quality_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size

        # 编码器
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        
        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        
        # 回报嵌入
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        
        # 动作嵌入 - 直接表示每个tile的质量
        self.embed_action = nn.Linear(tile_num, plm_embed_size).to(device)
        
        # 状态特征嵌入
        self.embed_state = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        
        # 层归一化
        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        
        # 动作头 - 预测每个tile的质量级别
        self.action_head = nn.Linear(plm_embed_size, tile_num * quality_levels).to(device)

        self.device = device
        self.device_out = device_out

        # 用于评估的队列
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        
        self.residual = residual
        self.which_layer = which_layer
        
        # 除PLM外的所有模块
        self.modules_except_plm = nn.ModuleList([  
            self.state_encoder, self.embed_timestep, self.embed_return, 
            self.embed_action, self.embed_state, self.embed_ln, self.action_head
        ])

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """前向传播函数，用于训练"""
        assert states.shape[0] == 1, '批量大小应为1以避免CUDA内存溢出'

        # 处理时间步和回报
        timesteps = timesteps.to(self.device)
        returns = returns.to(self.device)
        time_embeddings = self.embed_timestep(timesteps)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # 处理动作 - 每个tile的质量级别
        actions = actions.to(self.device)
        action_embeddings = self.embed_action(actions) + time_embeddings

        # 处理状态
        states = states.to(self.device)
        state_features = self.state_encoder(states)
        state_embeddings = self.embed_state(state_features) + time_embeddings
        
        # 堆叠所有嵌入
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1], dtype=int)
        
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((
                returns_embeddings[0, i:i+1], 
                state_embeddings[0, i:i+1],
                action_embeddings[0, i:i+1]
            ), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = i * 3 + 2
            
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  
        stacked_inputs_ln = self.embed_ln(stacked_inputs)
        
        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), 
                                      dtype=torch.long, device=self.device)

        # 使用预训练语言模型
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        
        if self.residual:
            logits = logits + stacked_inputs_ln
        
        # 预测动作
        action_logits = logits[:, action_embed_positions]
        action_pred = self.action_head(action_logits)
        
        return action_pred.view(-1, self.tile_num, self.quality_levels)

    def sample(self, state, target_return, timestep, **kwargs):
        """
        采样动作函数，直接预测所有tile的质量级别
        
        返回:
            quality_levels: 所有tile的质量级别决策
        """
        # 确定是否在训练模式
        exploration_mode = self.training
        
        # 衰减的epsilon值
        epsilon = max(0.05, 0.5 * (0.99 ** timestep)) if exploration_mode else 0.0
        
        # 以epsilon的概率随机探索
        if random.random() < epsilon:
            # 随机质量级别，使用平滑分布
            quality_levels = np.zeros(self.tile_num, dtype=int)
            # 倾向于更高的质量水平
            probabilities = np.array([0.1, 0.2, 0.3, 0.4])  # 质量级别0-3的概率
            
            for i in range(self.tile_num):
                quality_levels[i] = np.random.choice(self.quality_levels, p=probabilities)
            
            # 更新历史队列
            quality_norm = quality_levels / (self.quality_levels - 1)
            self._update_action_history(quality_norm, target_return, timestep)
            
            return quality_levels
        
        # 堆叠之前的状态、动作、回报特征
        prev_stacked_inputs = []
        for i in range(len(self.states_dq)):
            prev_return_embeddings = self.returns_dq[i]
            prev_state_embeddings = self.states_dq[i]
            prev_action_embeddings = self.actions_dq[i]
            
            prev_stacked_inputs.append(torch.cat((
                prev_return_embeddings, 
                prev_state_embeddings, 
                prev_action_embeddings
            ), dim=1))
            
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # 处理目标回报和时间步
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)
        return_embeddings = return_embeddings + time_embeddings

        # 处理状态
        state = state.to(self.device)
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        state_features = self.state_encoder(state)
        state_embeddings = self.embed_state(state_features) + time_embeddings

        # 堆叠回报、状态和之前的嵌入
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)
        
        # 截断序列长度
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]
        stacked_inputs_ln = self.embed_ln(stacked_inputs)
        
        # 创建注意力掩码
        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), 
                                  dtype=torch.long, device=self.device)

        # 使用预训练语言模型
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        
        if self.residual:
            logits = logits + stacked_inputs_ln
        
        # 预测质量级别
        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used).view(1, self.tile_num, self.quality_levels)
        # 为每个tile选择质量级别
        quality_probs = F.softmax(action_pred.squeeze(0), dim=1).cpu().numpy()
        quality_levels = np.zeros(self.tile_num, dtype=int)
        
        for i in range(self.tile_num):
            # if exploration_mode:
            #     # 训练模式下添加随机性
            #     temperature = max(0.5, 1.0 * (0.99 ** timestep))
            #     adjusted_probs = np.power(quality_probs[i], 1.0/temperature)
            #     adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
            #     quality_levels[i] = np.random.choice(self.quality_levels, p=adjusted_probs)
            # else:
            #     quality_levels[i] = np.argmax(quality_probs[i])
            quality_levels[i] = np.argmax(quality_probs[i])

        # 更新历史
        quality_norm = quality_levels / (self.quality_levels - 1)
        self._update_action_history(quality_norm, target_return, timestep)
        
        return quality_levels
    
    def _update_action_history(self, quality_norm, target_return, timestep):
        """更新历史队列"""
        action_tensor = torch.tensor(quality_norm, dtype=torch.float32, 
                                   device=self.device).reshape(1, 1, self.tile_num)
        
        time_embeddings = self.embed_timestep(
            torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)
        )
        
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        
        return_embeddings = self.embed_return(
            torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        ) + time_embeddings
        
        state_embeddings = self.states_dq[-1] if self.states_dq else time_embeddings
        
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings)
        self.actions_dq.append(action_embeddings)
        
    def clear_dq(self):
        """清空双端队列并重新初始化"""
        self.states_dq.clear()
        self.returns_dq.clear()
        self.actions_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))