"""
Customized state encoder for point cloud video streaming.
"""
import torch
import torch.nn as nn
from model_pcv.Hyperparameters import TILE_IN_F,QUALITY_LEVELS

class EncoderNetwork(nn.Module):
    """
    The encoder network for encoding point cloud streaming state information.
    Modified from Pensieve's encoder to adapt to point cloud video features.
    """
    def __init__(self, conv_size=4,enbed_dim=128):
        """
        初始化点云视频编码器网络。

        Args:
            `enbed_dim`: 嵌入维度
            
        """
        super().__init__()
        self.enbed_dim = enbed_dim
        self.conv_size = conv_size
        
        # 编码网络带宽特征
        self.cov_bandwidth = nn.Sequential(nn.Conv1d(1, enbed_dim, conv_size), nn.LeakyReLU(),nn.Flatten())  
        
        # 编码FOV历史特征
        self.conv_fov = nn.Sequential(
            nn.Conv1d(1, enbed_dim // 2, kernel_size=4, stride=2),  # 第一层卷积
            nn.LeakyReLU(),
            nn.Conv1d(enbed_dim // 2, enbed_dim, kernel_size=4, stride=2),  # 第二层卷积
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),  # 自适应平均池化到固定维度
            nn.Flatten()  # 展平输出
        )
        #self.fc_fov = nn.Sequential(nn.Linear(50 * TILE_IN_F, enbed_dim), nn.LeakyReLU())  
        
        # 编码缓冲区特征
        self.fc_buffer = nn.Sequential(nn.Linear(1 , enbed_dim), nn.LeakyReLU())  
        
        # # 编码GOF大小特征
        # self.fc_gof_size = nn.Sequential(
        #     nn.Linear(TILE_IN_F * QUALITY_LEVELS, enbed_dim), 
        #     nn.LeakyReLU()
        # )
        
        # # 编码视频帧计数器
        # self.fc_frame_counter = nn.Sequential(nn.Linear(1, enbed_dim), nn.LeakyReLU())
        
        # 输出层 - 将特征融合到输出维度
        self.output_layer = nn.Sequential(
            nn.Linear(enbed_dim * 3, enbed_dim),
            nn.LayerNorm(enbed_dim)
        )


    def forward(self, state):
        """
        前向传播函数。

        Args:
            state: 输入状态张量,包含点云流媒体相关特征
            state 的形状是 [1, 10, 672],这是一个3D张量
            表示 [batch_size, sequence_length, feature_dim]
        Returns:
            encoded_state: 编码后的状态特征
        """
        #print(f"state shape: {state.shape}")
        batch_size, seq_len, feature_dim = state.shape
        # 在时间维度上平均，将3D张量转为2D
        state_avg = torch.mean(state, dim=1)  # [batch_size, feature_dim]
        # 解析输入状态的不同部分
        bandwidth_features = state_avg[:, :10]  # 带宽特征 [batch_size, 10]
        #print(f"bandwidth_features shape: {bandwidth_features.shape}")
        
        # 检查Linear层的输入维度
        #print(f"fc_bandwidth input dim: {self.fc_bandwidth[0].in_features}")
        #print(f"fc_bandwidth weight shape: {self.fc_bandwidth[0].weight.shape}")
        
        # FOV历史特征 (假设历史FOV是50帧,每帧TILE_IN_F个tile)
        fov_dim = 50 * TILE_IN_F
        fov_features = state[:, 10:10+fov_dim]
        fov_features = fov_features.unsqueeze(1)  # 添加通道维度 [batch_size, 1, sequence_length]
        
        # 缓冲区特征 (缓冲区大小和缓冲区状态)
        buffer_size = state[:, 10+fov_dim:10+fov_dim+1]
        buffer_state = state[:, 10+fov_dim+1:10+fov_dim+1+TILE_IN_F]
        buffer_features = torch.cat([buffer_size, buffer_state], dim=1)
        
        # GOF大小特征
        # gof_size_dim = TILE_IN_F * QUALITY_LEVELS
        # gof_size_features = state[:, 10+fov_dim+1+TILE_IN_F:10+fov_dim+1+TILE_IN_F+gof_size_dim]
        
        # 视频帧计数器
        #frame_counter = state[:, -1:] 
        
        # 编码各个特征
        encoded_bandwidth = self.cov_bandwidth(bandwidth_features)
        encoded_fov = self.conv_fov(fov_features)
        encoded_buffer = self.fc_buffer(buffer_features)
        # encoded_gof_size = self.fc_gof_size(gof_size_features)
        # encoded_frame = self.fc_frame_counter(frame_counter)
        
        # 特征融合
        combined_features = torch.cat([
            encoded_bandwidth, 
            encoded_fov, 
            encoded_buffer,
            # encoded_gof_size,
            # encoded_frame
        ], dim=1)
        
        # 输出编码后的状态
        encoded_state = self.output_layer(combined_features)
        
        return encoded_state