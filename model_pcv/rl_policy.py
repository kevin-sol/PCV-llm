import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils.logger import setup_logger

class PointCloudRLPolicy(nn.Module):
    """
    为点云流媒体设计的强化学习策略网络。
    这个网络结合了预训练语言模型(Llama-2-7B)和点云状态编码器。
    """
    def __init__(
            self,
            state_feature_dim,      # 状态特征维度
            tile_num,               # tile数量
            quality_levels,         # 质量级别数
            state_encoder,          # 状态编码器
            plm,                    # 预训练语言模型
            plm_embed_size,         # PLM嵌入维度
            max_length=50,        # 最大序列长度
            max_ep_len=10,         # 最大回合长度
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

        self.plm = plm  # Llama-2-7B 
        self.plm_embed_size = plm_embed_size

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        
        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        
        # 回报嵌入
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        
        # tile选择嵌入 (新增)
        self.embed_tile_selection = nn.Linear(tile_num, plm_embed_size).to(device)
        
        # 质量级别嵌入 (修改)
        self.embed_quality = nn.Linear(tile_num, plm_embed_size).to(device)
        
        # 状态特征嵌入
        self.embed_state = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        
        # 层归一化
        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        # =========== multimodal encoder (end) ===========
    
        # 动作头部网络,针对点云环境进行修改
        # 分为两个部分：tile选择头和质量级别头
        self.tile_selection_head = nn.Linear(plm_embed_size, tile_num).to(device)  
        self.quality_head = nn.Linear(plm_embed_size, tile_num * quality_levels).to(device)  

        self.device = device
        self.device_out = device_out

        # 用于评估的队列
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.tile_selections_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.qualities_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer
        
        # 除PLM外的所有模块,用于保存和加载
        self.modules_except_plm = nn.ModuleList([  
            self.state_encoder, self.embed_timestep, self.embed_return, 
            self.embed_tile_selection, self.embed_quality, self.embed_state, 
            self.embed_ln, self.tile_selection_head, self.quality_head
        ])

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        前向传播函数,用于训练。
        
        Args:
            states: 状态序列
            actions: 动作序列 (包含tile选择和质量级别)
            returns: 回报序列
            timesteps: 时间步序列
            attention_mask: 注意力掩码
            
        Returns:
            tile_selection_pred: 预测的tile选择
            quality_pred: 预测的质量级别
        """
        assert states.shape[0] == 1, '批量大小应为1以避免CUDA内存溢出'

        # 处理时间步和回报
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns) + time_embeddings

        # 处理动作 - 分离tile选择和质量级别
        actions = actions.to(self.device)  # shape: (1, seq_len, tile_num+tile_num)
        tile_selections = actions[:, :, :self.tile_num]  # 前半部分是tile选择
        quality_levels = actions[:, :, self.tile_num:]  # 后半部分是质量级别
        
        # 嵌入tile选择和质量级别
        tile_selection_embeddings = self.embed_tile_selection(tile_selections) + time_embeddings
        quality_embeddings = self.embed_quality(quality_levels) + time_embeddings

        # 处理状态 - 使用自定义的状态编码器
        states = states.to(self.device)
        state_features = self.state_encoder(states)
        state_embeddings = self.embed_state(state_features) + time_embeddings
        
        # 堆叠所有嵌入
        # 序列结构: (R_1, s_1, tile_1, quality_1, R_2, s_2, tile_2, quality_2, ...)
        stacked_inputs = []
        action_embed_positions = np.zeros((returns_embeddings.shape[1], 2))  # 记录tile选择和质量级别嵌入的位置
        
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((
                returns_embeddings[0, i:i+1], 
                state_embeddings[0, i:i+1],
                tile_selection_embeddings[0, i:i+1], 
                quality_embeddings[0, i:i+1]
            ), dim=0)
            stacked_inputs.append(stacked_input)
            # 记录动作嵌入位置
            action_embed_positions[i, 0] = (i + 1) * 4 - 2  # tile选择位置
            action_embed_positions[i, 1] = (i + 1) * 4 - 1  # 质量级别位置
            
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        # 截断序列长度
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # 层归一化
        
        # 创建注意力掩码
        if attention_mask is None:
            # 1表示可以被注意到,0表示不能
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # 使用预训练语言模型处理嵌入序列
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        
        if self.residual:
            logits = logits + stacked_inputs_ln  # 残差连接
        
        # 预测tile选择和质量级别
        # 获取对应位置的logits
        tile_selection_logits = logits[:, action_embed_positions[:, 0].astype(int)]
        quality_logits = logits[:, action_embed_positions[:, 1].astype(int)]
        
        # 应用预测头
        tile_selection_pred = self.tile_selection_head(tile_selection_logits)
        quality_pred = self.quality_head(quality_logits)
        
        return tile_selection_pred, quality_pred.view(-1, self.tile_num, self.quality_levels)

    def sample(self, state, target_return, timestep, **kwargs):
        """
        采样动作函数,用于评估/测试。
        
        Args:
            state: 当前状态
            target_return: 目标回报
            timestep: 当前时间步
            
        Returns:
            selected_tile: 选择的tile
            selected_quality: 选择的质量级别
        """
        # 堆叠之前的状态、动作、回报特征
        prev_stacked_inputs = []
        for i in range(len(self.states_dq)):
            prev_return_embeddings = self.returns_dq[i]
            prev_state_embeddings = self.states_dq[i]
            prev_tile_embeddings = self.tile_selections_dq[i]
            prev_quality_embeddings = self.qualities_dq[i]
            
            prev_stacked_inputs.append(torch.cat((
                prev_return_embeddings, 
                prev_state_embeddings, 
                prev_tile_embeddings,
                prev_quality_embeddings
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
        #处理输入的变量
        #这是因为 preprocess_state处理后state的维度是2
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # 添加序列维度
        state_features = self.state_encoder(state)
        state_embeddings = self.embed_state(state_features) + time_embeddings

        # 堆叠回报、状态和之前的嵌入
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)
        
        # 截断序列长度
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]
        stacked_inputs_ln = self.embed_ln(stacked_inputs)
        
        # 创建注意力掩码
        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # 使用预训练语言模型处理
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        
        if self.residual:
            logits = logits + stacked_inputs_ln
        
        # 预测tile选择和质量级别
        logits_used = logits[:, -1:]
        tile_selection_pred = self.tile_selection_head(logits_used)
        quality_pred = self.quality_head(logits_used).view(1, self.tile_num, self.quality_levels)
        
        # 应用sigmoid到tile选择预测
        tile_probs = torch.sigmoid(tile_selection_pred).squeeze(0).squeeze(0)
        #self.logger.info(f"tile_probs: {tile_probs}")
        # 增加探索 - 在训练阶段添加随机性
        # if self.training:
        #     exploration_prob = 0.2  # 20% 的随机探索概率
        #     random_mask = torch.rand(tile_probs.size()) < exploration_prob
        #     random_values = torch.rand(tile_probs.size())
        #     tile_probs = torch.where(random_mask.to(self.device), random_values.to(self.device), tile_probs)
        selected_tile = (tile_probs > 0.3).float().cpu().numpy()
        #self.logger.info(f"after explore tile_probs: {tile_probs}")
        # 确保至少选择8个tile
        selected_count = np.sum(selected_tile)
        if selected_count < 8:
            # 获取未选中tile的概率排序
            unselected_indices = np.where(selected_tile == 0)[0]
            unselected_probs = tile_probs[unselected_indices]
            # 选择概率最高的(6-selected_count)个tile
            num_to_select = 8 - int(selected_count)
            top_indices = torch.topk(torch.tensor(unselected_probs), num_to_select).indices.cpu().numpy()
            selected_tile[unselected_indices[top_indices]] = 1.0
        
        # 为每个选中的tile选择质量级别
        quality_probs = F.softmax(quality_pred.squeeze(0), dim=1).cpu().numpy()
        selected_quality = np.zeros(self.tile_num, dtype=int)
        
        for i in range(self.tile_num):
            if selected_tile[i] > 0.1:  # 如果tile被选中 
                selected_quality[i] = np.argmax(quality_probs[i])
        #self.logger.info(f"selected_tile: {selected_tile}")
        #self.logger.info(f"selected_quality: {selected_quality}")
        # 计算动作嵌入
        tile_tensor = torch.as_tensor(selected_tile, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        quality_tensor = torch.zeros(1, 1, self.tile_num, dtype=torch.float32, device=self.device)
        for i in range(self.tile_num):
            if selected_tile[i] > 0.1:
                quality_tensor[0, 0, i] = selected_quality[i] / (self.quality_levels - 1)
        
        # 嵌入动作
        tile_embeddings = self.embed_tile_selection(tile_tensor) + time_embeddings
        quality_embeddings = self.embed_quality(quality_tensor) + time_embeddings
        
        # 更新队列
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings)
        self.tile_selections_dq.append(tile_embeddings)
        self.qualities_dq.append(quality_embeddings)
        #self.logger.info(f"selected_tile: {selected_tile}")
        #self.logger.info(f"selected_quality: {selected_quality}")
        return selected_tile, selected_quality
    
    def clear_dq(self):
        """清空双端队列并重新初始化"""
        self.states_dq.clear()
        self.returns_dq.clear()
        self.tile_selections_dq.clear()
        self.qualities_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.tile_selections_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.qualities_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))