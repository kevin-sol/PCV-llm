import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model_pcv.rl_policy import PointCloudRLPolicy
from model_pcv.state_encoder import EncoderNetwork
from utils.model import save_model, load_model
from model_pcv.config import cfg
from model_pcv.Hyperparameters import TILE_IN_F,F_IN_GOF,QUALITY_LEVELS,VIDEO_GOF_LEN,\
    FRAME,INIT_QOE,REBUF_PENALTY,SMOOTH_PENALTY,MULTIPLE_QUALITY_LEVELS
from utils.logger import setup_logger
from model_pcv.PrioritizedBuffer import PrioritizedBuffer

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
    
    # 创建用于保存损失的目录
    logs_dir = os.path.join(model_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 创建损失日志文件
    loss_log_path = os.path.join(logs_dir, 'loss_log.txt')
    with open(loss_log_path, 'w') as f:
        f.write("Episode,Loss,Reward,Quality,Rebuffer,Switch\n")  # 写入标题行
    
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
            # 每个 episode 的统计数据（汇总所有 FOV 轨迹）
        episode_rewards_all = []
        episode_qualities_all = []
        episode_rebuffers_all = []
        episode_switches_all = []
        epoch_losses = []
        
        # 遍历所有FOV轨迹
        for fov_trace_idx in range(len(fov_traces)):
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
                switch_penalty=switch*SMOOTH_PENALTY
                reward = quality_reward - rebuffer_penalty-switch_penalty
                episode_returns.append(reward)
                
                # 记录当前步骤
                episode_timesteps.append(gof_index)
                
                # 更新统计
                # 一个视频中每一个gof的
                episode_qualities.append(quality)
                episode_rebuffers.append(rebuffer)
                episode_switch.append(switch)
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
                        # 记录每批次的平均损失
                    avg_batch_loss = accumulated_loss / gradient_accumulation_steps
                    epoch_losses.append(avg_batch_loss)
                    self.stats['losses'].append(avg_batch_loss)
                    
            # 计算每个视频的gof总和
            sum_reward = np.sum(episode_reward)
            sum_quality = np.sum(episode_qualities) 
            sum_rebuffer = np.sum(episode_rebuffers)
            sum_switch = np.sum(episode_switch)
            
            episode_rewards_all.append(sum_reward)
            episode_qualities_all.append(sum_quality)
            episode_rebuffers_all.append(sum_rebuffer)
            episode_switches_all.append(sum_switch)
        
        # for循环结束
        avg_epoch_reward = np.mean(episode_rewards_all)
        avg_epoch_quality = np.mean(episode_qualities_all)
        avg_epoch_rebuffer = np.mean(episode_rebuffers_all)
        avg_epoch_switch = np.mean(episode_switches_all)
        avg_epoch_loss = np.mean(epoch_losses) 
        # 存储统计信息到全局统计
        self.stats['episode_rewards'].append(avg_epoch_reward)
        self.stats['episode_qualities'].append(avg_epoch_quality)
        self.stats['episode_rebuffers'].append(avg_epoch_rebuffer)
        self.stats['episode_switch'].append(avg_epoch_switch)
        # 记录损失到文件
        with open(loss_log_path, 'a') as f:
            f.write(f"{episode},{avg_epoch_loss:.8f},{avg_epoch_reward:.2f},{avg_epoch_quality:.2f},{avg_epoch_rebuffer:.4f},{avg_epoch_switch:.4f}\n")
        
        # 报告训练进度（每个epoch只打印一次）
        if episode % report_interval == 0:
            print(f"--------Epoch {episode}/{num_episodes}--------")
            print(f"Avg Reward: {avg_epoch_reward:.4f} | "
                f"Avg Quality: {avg_epoch_quality:.4f} | "
                f"Avg Rebuffer: {avg_epoch_rebuffer:.4f}s | "
                f"Avg Switch: {avg_epoch_switch:.4f} | "
                f"Avg Loss: {avg_epoch_loss:.6f}")
            
        # 每隔save_interval保存模型检查点
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_ep_{episode}")
            save_model(self.args, self.policy, checkpoint_path)
            print(f"模型检查点已保存: {checkpoint_path}")

        # 每隔eval_interval评估模型
        if episode % eval_interval == 0 and episode > 0:
            print(f"\n开始第 {episode} 轮评估...")
            eval_reward, eval_quality, eval_rebuffer, eval_switch = self.evaluate(fov_traces, dis_traces)

            # 记录评估结果
            self.stats['eval_rewards'].append(eval_reward)
            self.stats['eval_qualities'].append(eval_quality)
            self.stats['eval_rebuffers'].append(eval_rebuffer)
            self.stats['eval_switch'].append(eval_switch)
            print(f"以整个视频为单位的评估结果:")
            print(f"平均回报: {eval_reward:.4f}, "
                    f"平均质量: {eval_quality:.4f}, "
                    f"平均重缓冲: {eval_rebuffer:.4f}s, "
                    f"平均切换: {eval_switch:.4f}")
            # 如果当前模型性能更好，保存为最佳模型
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_model(self.args, self.policy, best_model_dir)
                print(f"新的最佳模型已保存: {best_model_dir}")
            print("")
            
    # 训练结束，保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    save_model(self.args, self.policy, final_model_path)
    print(f"最终模型已保存: {final_model_path}")
    
    return self.stats