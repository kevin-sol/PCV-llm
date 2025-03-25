import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model_pcv.rl_policy import PointCloudRLPolicy
from model_pcv.state_encoder import EncoderNetwork
from utils.plm_utils import load_plm
from utils.model import save_model, load_model
from model_pcv.config import cfg
from model_pcv.low_rank import peft_model
from model_pcv.Hyperparameters import TILE_IN_F,F_IN_GOF,QUALITY_LEVELS,VIDEO_GOF_LEN,\
    FRAME,INIT_QOE,REBUF_PENALTY,SMOOTH_PENALTY,MULTIPLE_QUALITY_LEVELS
from utils.logger import setup_logger
from model_pcv.PrioritizedBuffer import PrioritizedBuffer
class LlamaPointCloudController:
    """使用Llama-2-7B控制点云流媒体系统"""
    
    def __init__(self, args, env, video_size,device='cuda'):
        """
        初始化控制器
        
        参数:
            args: 模型参数
            env: 点云环境
            device: 运行设备
        """
        self.logger = setup_logger('train_logs')
        
        self.playback_pos=0
        self.history_len=50
        
        self.args = args
        self.env = env
        self.device = device
        self.video_size = video_size
        self.stats = {
        'episode_rewards': [],
        'episode_qualities': [],
        'episode_rebuffers': [],
        'episode_switch': [],
        'losses': [],
        'eval_rewards': [],
        'eval_qualities': [],
        'eval_rebuffers': [],
        'eval_switch': [],
    }
        # 计算状态维度
        state_dim = self._get_state_dim()
        
        # 加载Llama-2-7B模型，支持模型并行
        self.plm, *_ = load_plm(
            args.plm_type, 
            os.path.join(args.plm_dir, args.plm_type, args.plm_size),  # 基于本地下载的llama-2-7B
            device_input_side=args.device, 
            device_output_side=args.device_out, 
            device_middle_side=args.device_mid
        ) 

        if args.plm_type != 'llama':
            self.plm = self.plm.to(args.device)
        
        # 使用PEFT进行低秩适应
        if args.rank != -1:
            self.plm = peft_model(self.plm, args.plm_type, rank=args.rank)
        
        # 创建状态编码器
        assert args.state_feature_dim is not None, 'please specify state feature dim to create state encoder'
        self.state_encoder = EncoderNetwork(
            # input_dim=state_dim,
            # hidden_dim=args.hidden_dim,
            embed_dim=args.state_feature_dim
        ).to(device)
        
        # 创建策略网络
        self.policy = PointCloudRLPolicy(
            state_feature_dim=args.state_feature_dim,
            tile_num=TILE_IN_F,
            quality_levels=QUALITY_LEVELS,
            state_encoder=self.state_encoder,
            plm=self.plm,
            plm_embed_size=args.plm_embed_size,
            max_length=args.max_length,
            max_ep_len=args.max_ep_len,
            device=device,
            device_out=args.device_out,
            residual=False,
            which_layer=args.which_layer,
            #conv_size=4
        )
        
        # 创建优化器
        # self.optimizer = AdamW(self.policy.modules_except_plm.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        self.optimizer = AdamW(self.policy.parameters(), lr=args.lr,weight_decay=args.weight_decay)
       
        # 如果使用低秩适应，也包括这些参数
        # if args.rank != -1:
        #     self.optimizer.add_param_group({'params': self.plm.parameters()})
        
        # 初始化FOV历史缓冲区
        self.fov_history = []
        self.history_length = 50  # 记录过去50帧的FOV
    
    def _get_state_dim(self):
        """计算状态维度"""
        # 包括带宽特征、FOV历史、缓冲区信息等
        bandwidth_dim = 10  # 当前和历史带宽
        fov_history_dim = 50 * TILE_IN_F  # 50帧历史FOV
        buffer_dim = 1 + TILE_IN_F  # 缓冲区大小和当前GOF状态
        gof_size_dim = TILE_IN_F * QUALITY_LEVELS  # 下一个GOF的大小信息
        frame_counter_dim = 1  # 视频帧计数器
        
        return bandwidth_dim + fov_history_dim + buffer_dim + gof_size_dim + frame_counter_dim
    
    def update_fov_history(self, current_fov):
        """更新FOV历史"""
        self.fov_history.append(current_fov)
        if len(self.fov_history) > self.history_length:
            self.fov_history.pop(0)
    
    def preprocess_state(self):
        """预处理环境状态为张量格式"""
        # 1. 带宽特征
        bandwidth_features = []
        #形状1*3：[当前带宽，过去两个带宽]
        # 当前带宽
        current_bw = self.env.cooked_bw[self.env.mahimahi_ptr]  # 当前带宽 
        bandwidth_features.append(current_bw / max(self.env.cooked_bw[self.env.mahimahi_ptr:self.env.mahimahi_ptr+10]))  # 归一化
        
        # 历史带宽
        ptr = self.env.mahimahi_ptr
        for i in range(2):  # 过去2个带宽样本
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
            #fov_features.extend(self.fov_history[0])
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
        buffer_features.append(self.env.buffer_size )  # 秒
        
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
        # buffer_features的形状应该是 [buffersize,12个质量等级]
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
                    gof_size_features.append(tile_size )  # 为Mb
                    # gof_size_features形状12*4：[第一个tile的四个质量等级对应的size，第二个....]
        else:
            # 已经是最后一个GOF，填充0
            gof_size_features.extend([0.0] * (TILE_IN_F * QUALITY_LEVELS))
        
        # 5. 视频帧计数器
        frame_counter = [self.env.video_frame_counter / len(self.video_size)]  # 归一化
        
        # 组合所有特征
        state = np.concatenate([
            bandwidth_features,
            fov_features,
            buffer_features,
            gof_size_features,
            frame_counter
        ])
        #self.logger.info(f"state-shape: {state.shape}")
        #return torch.FloatTensor(state).reshape(1, 1, -1)
        return torch.FloatTensor(state).unsqueeze(0)
    
    def predict_fov_and_bitrate(self, current_frame, playback_pos,user_fov_trace):
        """
        使用Llama-2-7B预测FOV和比特率
        
        参数:
            current_frame: 当前帧索引
            playback_position: 当前播放位置（已知FOV的最后一帧）
            user_fov_trace: 用户FOV轨迹
            
        返回:
            selected_tile: 选择的tile
            selected_quality: 选择的质量级别
        """
        # 根据播放位置更新FOV历史（只使用已经播放过的帧的FOV）
        self.fov_history = []  # 清空历史
        for i in range(max(0, playback_pos - self.history_length), playback_pos):
            if i < len(user_fov_trace):
                self.fov_history.append(user_fov_trace[i])
    
        # 如果没有足够的历史FOV，使用默认策略
        if len(self.fov_history) < 50:  # 至少需要50帧历史
            selected_tile = np.ones(TILE_IN_F)  # 保守策略：选择所有tile
            selected_quality = np.zeros(TILE_IN_F, dtype=int)  # 默认最低质量
            return selected_tile, selected_quality
        
        # 预处理当前状态（只基于已知的FOV历史）
        state = self.preprocess_state()
        #此时state的shape是[1,665]

        # 使用策略网络预测动作
        with torch.no_grad():
            # 计算理论最优回报
            quality = 0
            for i in range(F_IN_GOF):
                for j in range(TILE_IN_F):
                    quality+=self.video_size[0+i][j][3]
            max_quality_reward = INIT_QOE * quality  # 最高质量奖励
            min_rebuffer = 0  # 理想情况无重缓冲
            min_switch = 0  # 理想情况无质量切换
            #target_return = max_quality_reward - min_rebuffer - min_switch
            
            # 计算目标回报
            if len(self.stats['episode_rewards']) > 0:
                # 使用历史最佳回报作为目标
                best_reward = max(self.stats['episode_rewards'])
                # 或使用最近N个episode的平均回报
                recent_avg = np.mean(self.stats['episode_rewards'][-10:])
                # 设置稍高于当前表现的目标
                target_return = max(best_reward, recent_avg) * 1.1
            else:
                # 初始阶段使用一个合理的估计值
                target_return = max_quality_reward   # 假设理想情况下每个GOF都能获得最高质量
            
            # 当前时间步（归一化）
            timestep = self.env.video_frame_counter // F_IN_GOF
            
            # 采样动作
            selected_tile, selected_quality = self.policy.sample(
                state=state,
                target_return=target_return,
                timestep=timestep
            )
        
        return selected_tile, selected_quality
    
    def stream_next_gof(self, user_fov_trace,user_dis_trace ,current_frame):
        """
        流式传输下一个GOF
        
        参数:
            user_fov_trace: 用户FOV轨迹
            current_frame: 当前帧索引
            
        返回:
            delay: 延迟
            rebuffer: 重缓冲时间
            quality: 平均质量
        """
        self.playback_pos=int(self.env.playback_position)
        # self.logger.info(f"播放位置：{self.playback_pos}")
        # 使用Llama-2-7B预测tile选择和质量级别
        selected_tile, selected_quality = self.predict_fov_and_bitrate(
            current_frame, 
            self.playback_pos,
            user_fov_trace
        )
        # 记录预测结果
        #tile_indices = [i for i, val in enumerate(selected_tile) if val > 0.1]
        #self.logger.info(f"预测的 tile: {tile_indices}")
        #selected_qualities = [selected_quality[i] for i in tile_indices]
        #self.logger.info(f"预测的质量: {selected_qualities}")
        
        # 执行下载
        delay, sleep_time, buffer_size, rebuffer, gof_size, done, gof_remain, buffer = \
            self.env.get_video_gof(selected_tile, selected_quality)
                
        # 计算平均质量
        seen = [0]*TILE_IN_F
        quality = 0
        #在真实fov中只要当前gof中有一帧的tile i为1，seen[i]=1
        for i in range(TILE_IN_F):
            for f in range(current_frame,current_frame+F_IN_GOF):
                if user_fov_trace[f][i]:
                    seen[i]=1
                    break
        for i in range(F_IN_GOF):
            for j in range(TILE_IN_F):
                if seen[j]*selected_tile[j]:
                    quality+=self.video_size[current_frame+i][j][selected_quality[j]]/user_dis_trace[current_frame+i][j]
        # 计算switch
        switch=0.0
        if(current_frame>0):
            for s in range(TILE_IN_F):
                switch+= \
                    np.abs(MULTIPLE_QUALITY_LEVELS[selected_quality[s]] \
                        -MULTIPLE_QUALITY_LEVELS[max(0,buffer[current_frame//F_IN_GOF-1][s])])
        return delay , rebuffer , quality,switch, selected_tile, selected_quality ,buffer # 为秒 
    
    
    def evaluate(self, fov_traces,dis_traces, num_traces=5):
        """
        评估模型在测试数据上的性能
        
        参数:
            fov_traces: FOV轨迹集合
            num_traces: 用于评估的轨迹数量
            
        返回:
            avg_reward: 平均回报
            avg_quality: 平均质量
            avg_rebuffer: 平均重缓冲时间
        """
        # 保存当前模型参数，以便评估后恢复
        # policy_state = {k: v.clone() for k, v in self.policy.modules_except_plm.state_dict().items()}
        # if self.args.rank != -1:
        #     plm_state = {k: v.clone() for k, v in self.plm.state_dict().items()}
        
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
                
                # 每个 gof 内所有帧使用相同的质量决策，所以这里保存上一 gof 的质量信息
                prev_quality = [0]*TILE_IN_F
                
                # 模拟视频播放（以 gof 为单位）
                current_frame = 0
                
                while current_frame < len(self.video_size):
                    delay, rebuffer, quality,switch, selected_tile, selected_quality ,buffer= self.stream_next_gof(
                        user_fov_trace, 
                        user_dis_trace,
                        current_frame
                    )
                    
                    # 计算 tile 级别的质量切换惩罚（按 gof 为单位）
                    # 这里 selected_quality 表示当前 gof 内每个 tile 的质量（均相同于该 gof 的决策）
                    switch_penalty = 0.0
                    for s in range(TILE_IN_F):
                        # 如果该 tile 被选择（例如 selected_tile[s] > 0.1 表示该 tile 有效）
                        switch_penalty+=SMOOTH_PENALTY *\
                            np.abs(MULTIPLE_QUALITY_LEVELS[max(buffer[current_frame//F_IN_GOF][s],0)]-MULTIPLE_QUALITY_LEVELS[prev_quality[s]])
            
                    # 计算质量奖励和重缓冲惩罚
                    quality_reward = quality * INIT_QOE
                    rebuffer_penalty = rebuffer * REBUF_PENALTY
                    reward = quality_reward - rebuffer_penalty - switch_penalty
                    
                    # 更新统计
                    # 一个视频中每一个gof的
                    trace_qualities.append(quality)
                    trace_rebuffer.append(rebuffer)
                    trace_switch.append(switch_penalty/SMOOTH_PENALTY)
                    trace_reward.append(reward)
                    # 更新上一帧的质量向量
                    prev_quality = selected_quality
                    
                    # 更新当前帧
                    current_frame += F_IN_GOF
                    
                    if current_frame >= len(self.video_size):
                        break
                
                # 记录轨迹结果
                # 平均每个视频的
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
        # 恢复模型参数
        # self.policy.modules_except_plm.load_state_dict(policy_state)
        # if self.args.rank != -1:
        #     self.plm.load_state_dict(plm_state)
        
        # 计算平均值
        # 测试的几个trace，平均每个trace
        avg_reward = np.mean(all_rewards)
        avg_quality = np.mean(all_qualities)
        avg_rebuffer = np.mean(all_rebuffers)
        avg_switch = np.mean(all_switches)
        return avg_reward, avg_quality, avg_rebuffer,avg_switch
    
    def train_step(self, states, actions, returns, timesteps, weights=None, update_params=True):
        """
        执行一个训练步骤
        
        参数:
            states: 状态批次
            actions: 动作批次（包含tile选择和质量级别）
            returns: 回报批次
            timesteps: 时间步批次
            weights: 样本权重，用于优先级经验回放
            update_params: 是否立即更新参数（用于梯度累积）
            
        返回:
            loss: 训练损失
            td_errors: TD误差，用于优先级更新
        """
        # 确保批次大小为1
        if update_params:
            self.optimizer.zero_grad()
        
        # 从策略网络获取预测
        #self.logger.info(f"train-step shape:{states.shape}")
        tile_selection_pred, quality_pred = self.policy(
            states=states,
            actions=actions,
            returns=returns,
            timesteps=timesteps
        )
        
        # 分离true标签
        tile_selection_true = actions[:, :, :TILE_IN_F]
        quality_true = actions[:, :, TILE_IN_F:]
        
        # 计算tile选择损失（二进制交叉熵）
        tile_loss = F.binary_cross_entropy_with_logits(
            tile_selection_pred,
            tile_selection_true
        )
        
        # 如果提供了权重，应用于损失函数
        if weights is not None:
            tile_loss = tile_loss * weights.view(-1, 1, 1)
        
        # 取平均
        tile_loss = tile_loss.mean()
        
        batch_size, seq_len, _ = tile_selection_true.shape
        # quality_true形状是1，10，12 batchsize,seqlen,tile 归一化质量级别
        # quality_pred形状是10，12，4 seqlen,tile,quality 概率
        # 计算质量级别损失
        quality_loss = 0
        quality_samples = 0
        quality_errors = []  # 存储每个样本的质量误差
        for b in range(batch_size):
            for s in range(seq_len):
                for t in range(TILE_IN_F):
                    if tile_selection_true[b, s, t] > 0.1:
                        # 将归一化的质量值转换为类别索引
                        target_quality = (quality_true[b, s, t] * (QUALITY_LEVELS- 1)).round().long()
                        
                        # 计算交叉熵损失
                        sample_loss = F.cross_entropy(
                            quality_pred[s, t].unsqueeze(0),
                            target_quality.unsqueeze(0),
                            reduction='none'
                        )
                        
                        # 如果提供了权重，应用于损失
                        if weights is not None:
                            sample_loss = sample_loss * weights[b]
                        
                        quality_loss += sample_loss
                        quality_samples += 1
                        quality_errors.append(sample_loss.item())

        # for s in range(seq_len):
        #     for t in range(TILE_IN_F):
        #         if tile_selection_true[0, s, t] > 0.1:  # batch_size=1，所以用索引0
        #             # 将归一化的质量值转换为类别索引
        #             target_quality = (quality_true[0, s, t] * (QUALITY_LEVELS - 1)).round().long()
                    
        #             # quality_pred[s, t] 形状为 [4]，表示当前tile的质量级别预测概率
        #             quality_loss += F.cross_entropy(
        #                 quality_pred[s, t].unsqueeze(0),  # 变为 [1, 4]
        #                 target_quality.unsqueeze(0)       # 变为 [1]
        #             )
        #             quality_samples += 1

        if quality_samples > 0:
            quality_loss = quality_loss / quality_samples
        else:
            print("警告: 没有有效的质量样本")
            quality_loss = torch.tensor(0.0, device=tile_loss.device)
        # 总损失
        loss = tile_loss + quality_loss
        # 计算TD误差（用于优先级更新）
        with torch.no_grad():
            # 简单地使用总损失作为TD误差的代理
            td_errors = torch.abs(loss) * torch.ones(batch_size, device=loss.device)
        
        # 反向传播
        if update_params:
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.modules_except_plm.parameters(), 5.0)
            
            # 如果使用低秩适应，也裁剪这些参数
            if self.args.rank != -1:
                torch.nn.utils.clip_grad_norm_(self.plm.parameters(), 5.0)
            
            # 更新参数
            self.optimizer.step()
        
        return loss ,td_errors
    
    def train(self, fov_traces,dis_traces, num_episodes=3000, batch_size=1, report_interval=10, 
          eval_interval=200, save_interval=200, model_dir=cfg.plm_ft_dir, gradient_accumulation_steps=4):
        """
        训练系统
        
        参数:
            fov_traces: FOV轨迹集合
            num_episodes: 训练的情节数
            batch_size: 批次大小
            report_interval: 报告间隔
            eval_interval: 评估间隔（每多少个episode评估一次）
            save_interval: 保存间隔（每多少个episode保存一次）
            model_dir: 模型保存目录
            
        返回:
            训练统计
        """
        
        # 创建模型保存目录
        best_model_dir = os.path.join(model_dir, 'best_model')
        checkpoint_dir = os.path.join(model_dir, 'checkpoints')
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 训练统计
        self.stats = {
        'episode_rewards': [],
        'episode_qualities': [],
        'episode_rebuffers': [],
        'episode_switch': [],
        'losses': [],
        'eval_rewards': [],
        'eval_qualities': [],
        'eval_rebuffers': [],
        'eval_switch': [],
    }
        # 记录最佳模型性能
        best_eval_reward = float('-inf')
        # 使用优先级经验回放缓冲区
        buffer_exp = PrioritizedBuffer(capacity=10000, alpha=0.6)
        # 优化器步骤计数
        optimization_step = 0

        for episode in range(num_episodes):
            
            # 随机选择一个FOV轨迹
            fov_trace_idx = np.random.randint(0, len(fov_traces))
            user_fov_trace = fov_traces[fov_trace_idx]
            user_dis_trace = dis_traces[fov_trace_idx]
            
            # 重置环境和历史
            self.env.reset()
            self.fov_history = []
            self.policy.clear_dq()
            
            # 情节统计
            episode_reward = []
            episode_qualities = []
            episode_rebuffers = []
            episode_switch = []
            
            
            # 本情节的经验
            episode_states = []
            episode_actions = []
            episode_returns = []
            episode_timesteps = []
            
            # 模拟视频播放
            current_frame = 0
            gof_index = 0
                        
            while current_frame < len(self.video_size):
                # 获取当前状态
                state = self.preprocess_state()
                #self.logger.info(f'train.shape: {state.shape}')
                episode_states.append(state)
                #记录当前帧
                #self.logger.info(f'当前帧: {current_frame}')
                # 流式传输下一个GOF
                delay, rebuffer, quality, switch, selected_tile, selected_quality,buffer = self.stream_next_gof(
                    user_fov_trace, 
                    user_dis_trace,
                    current_frame
                )
                # self.logger.info(f"selected_tile: {selected_tile}")
                # self.logger.info(f"selected_quality: {selected_quality}")
                # self.logger.info(f"switch: {switch}")
                #self.logger.info(f"delay: {delay}, rebuffer: {rebuffer}, quality: {quality}")
                 # 计算 tile 级别的质量切换惩罚（按 gof 为单位）
                switch_penalty = switch*SMOOTH_PENALTY
                # for s in range(TILE_IN_F):
                #     # 如果该 tile 被选择（例如 selected_tile[s] > 0.1 表示该 tile 有效）
                #     switch_penalty+=SMOOTH_PENALTY *\
                #         np.abs(MULTIPLE_QUALITY_LEVELS[max(buffer[current_frame//F_IN_GOF][s],0)]-MULTIPLE_QUALITY_LEVELS[prev_quality[s]])
                
                # 构造动作向量
                # 将 selected_quality 转换为 NumPy 数组后再进行除法操作
                action = np.concatenate([
                    selected_tile, 
                    np.array(selected_quality, dtype=np.float32) / (QUALITY_LEVELS - 1)
                ])
                #action = np.concatenate([selected_tile, selected_quality / (QUALITY_LEVELS - 1)])
                episode_actions.append(action)
                
                # 计算回报
                # 一个gof的回报
                quality_reward = quality * INIT_QOE
                rebuffer_penalty = rebuffer * REBUF_PENALTY
                switch_penalty=switch_penalty
                reward = quality_reward - rebuffer_penalty-switch_penalty
                episode_returns.append(reward)
                
                # 记录当前步骤
                episode_timesteps.append(gof_index)
                
                # 更新统计
                # 一个视频中每一个gof的
                episode_qualities.append(quality)
                episode_rebuffers.append(rebuffer)
                episode_switch.append(switch_penalty/SMOOTH_PENALTY)
                episode_reward.append(reward)
                
                # 更新当前帧和GOF索引
                current_frame += F_IN_GOF
                gof_index += 1
                
                # 如果视频结束，跳出循环
                if current_frame >= len(self.video_size):
                    break
            
            # 收集本情节的经验到缓冲区
            experience=(
                torch.cat(episode_states, dim=0),
                torch.tensor(episode_actions, dtype=torch.float32),
                torch.tensor(episode_returns, dtype=torch.float32).unsqueeze(1),
                torch.tensor(episode_timesteps, dtype=torch.int32)
            )
            # 使用平均回报作为初始优先级
            initial_priority = abs(np.mean(episode_returns)) + 1e-5
            buffer_exp.add(experience, initial_priority)
        
            # 如果缓冲区足够大，进行训练
            if buffer_exp.size >= batch_size:
                 # 准备累积梯度更新的参数
                self.optimizer.zero_grad()
                accumulated_loss = 0.0
                
                # 采样batch_size*gradient_accumulation_steps个样本
                effective_batch_size = batch_size * gradient_accumulation_steps
                if buffer_exp.size >= effective_batch_size:
                    # 分批处理，每批batch_size个样本
                    batch_samples, batch_indices, batch_weights = buffer_exp.sample(
                        effective_batch_size, beta=0.4
                    )
                    
                    # 将样本分成gradient_accumulation_steps批次
                    for i in range(0, effective_batch_size, batch_size):
                        end_idx = min(i + batch_size, effective_batch_size)
                        mini_batch = batch_samples[i:end_idx]
                        mini_batch_indices = batch_indices[i:end_idx]
                        mini_batch_weights = batch_weights[i:end_idx]
                        
                        # 解包小批量
                        batch_states, batch_actions, batch_returns, batch_timesteps = zip(*mini_batch)
                        
                        # 执行训练步骤但不更新参数
                        loss, td_errors = self.train_step(
                            states=torch.stack(batch_states).to(self.device),
                            actions=torch.stack(batch_actions).to(self.device),
                            returns=torch.stack(batch_returns).to(self.device),
                            timesteps=torch.stack(batch_timesteps).to(self.device),
                            weights=torch.tensor(mini_batch_weights, dtype=torch.float32).to(self.device),
                            update_params=False  # 不立即更新参数
                        )
                        
                        # 缩放损失以适应梯度累积
                        scaled_loss = loss / gradient_accumulation_steps
                        scaled_loss.backward()
                        
                        # 累积损失（用于日志记录）
                        accumulated_loss += loss.item()
                        
                        # 更新样本优先级
                        new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-5
                        buffer_exp.update_priorities(mini_batch_indices, new_priorities)
                    
                    # 完成梯度累积后更新参数
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.policy.modules_except_plm.parameters(), 5.0)
                    if self.args.rank != -1:
                        torch.nn.utils.clip_grad_norm_(self.plm.parameters(), 5.0)
                    
                    self.optimizer.step()
                    optimization_step += 1
                    
                    # 记录平均损失
                    self.stats['losses'].append(accumulated_loss / gradient_accumulation_steps)
                        
            # 记录情节统计
            # 平均每个gof的reward
            avg_reward = np.mean(episode_reward)
            avg_quality = np.mean(episode_qualities) if episode_qualities else 0
            avg_rebuffer = np.mean(episode_rebuffers)
            avg_switch = np.mean(episode_switch)
            # 计算每个视频的gof总和
            sum_reward = np.sum(episode_reward)
            sum_quality = np.sum(episode_qualities) if episode_qualities else 0
            sum_rebuffer = np.sum(episode_rebuffers)
            sum_switch = np.sum(episode_switch)
            
            self.stats['episode_rewards'].append(sum_reward)
            self.stats['episode_qualities'].append(sum_quality)
            self.stats['episode_rebuffers'].append(sum_rebuffer)
            self.stats['episode_switch'].append(sum_switch)
            
            # 报告训练进度
            if episode % report_interval == 0:
                avg_loss = np.mean(self.stats['losses'][-report_interval:]) if self.stats['losses'] else 0
                print("平均每个gof:")
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {sum_reward:.4f} | "
                      f"Quality: {sum_quality:.4f} | "
                      f"Rebuffer: {sum_rebuffer:.4f}s | "
                      f"switch: {sum_switch:.4f} |"
                      f"Loss: {avg_loss:.6f}")
            # 每隔save_interval保存模型检查点
            if episode % save_interval == 0 and episode > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_ep_{episode}")
                save_model(self.args, self.policy, checkpoint_path)
                # torch.save({
                #     'policy_state_dict': self.policy.modules_except_plm.state_dict(),
                #     'plm_state_dict': self.plm.state_dict() if self.args.rank != -1 else None,
                #     'optimizer_state_dict': self.optimizer.state_dict(),
                #     'episode': episode,
                #     'self.stats': self.stats
                # }, checkpoint_path)
                # self.policy.plm.save_pretrained(checkpoint_path)
                print(f"模型检查点已保存: {checkpoint_path}")

            # 每隔eval_interval评估模型
            if episode % eval_interval == 0 and episode > 0:
                print(f"\n开始第 {episode} 轮评估...")
                eval_reward, eval_quality, eval_rebuffer, eval_switch= self.evaluate(fov_traces,dis_traces)

                # 记录评估结果
                self.stats['eval_rewards'].append(eval_reward)
                self.stats['eval_qualities'].append(eval_quality)
                self.stats['eval_rebuffers'].append(eval_rebuffer)
                self.stats['episode_switch'].append(eval_switch)
                print(f"以整个视频为单位的评估结果:")
                print(f"平均回报: {eval_reward:.4f}, "
                        f"平均质量: {eval_quality:.4f}, "
                        f"平均重缓冲: {eval_rebuffer:.4f}s,"
                        f"平均切换: {eval_switch:.4f}")
                # 如果当前模型性能更好，保存为最佳模型
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    save_model(self.args, self.policy, best_model_dir)
                    # best_model_path = os.path.join(best_model_dir, "best_model.pth")
                    # torch.save({
                    #     'policy_state_dict': self.policy.modules_except_plm.state_dict(),
                    #     'plm_state_dict': self.plm.state_dict() if self.args.rank != -1 else None,
                    #     'optimizer_state_dict': self.optimizer.state_dict(),
                    #     'episode': episode,
                    #     'eval_reward': eval_reward,
                    #     'eval_quality': eval_quality,
                    #     'eval_rebuffer': eval_rebuffer,
                    #     'self.stats': self.stats
                    # }, best_model_path)
                    print(f"新的最佳模型已保存: {best_model_dir}")
                print("")
        # 训练结束，保存最终模型
        final_model_path = os.path.join(model_dir, "final_model")
        save_model(self.args, self.policy, final_model_path)
        # torch.save({
        #     'policy_state_dict': self.policy.modules_except_plm.state_dict(),
        #     'plm_state_dict': self.plm.state_dict() if self.args.rank != -1 else None,
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'episode': num_episodes,
        #     'self.stats': self.stats
        # }, final_model_path)
        print(f"最终模型已保存: {final_model_path}")
        
        return self.stats