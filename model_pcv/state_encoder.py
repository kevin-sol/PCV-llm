"""
Customized state encoder for point cloud video streaming.
"""
import torch
import torch.nn as nn
from model_pcv.Hyperparameters import TILE_IN_F, QUALITY_LEVELS

class EncoderNetwork(nn.Module):
    """
    The encoder network for encoding point cloud streaming state information.
    Modified from Pensieve's encoder to adapt to point cloud video features.
    
    处理形状为 [batch_size, seq_len, feature_dim] 的3D状态输入，
    使用卷积神经网络和全连接层提取特征。
    """
    def __init__(self, embed_dim=128):
        """
        初始化点云视频编码器网络。

        Args:
            embed_dim: 嵌入维度，默认为128
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 编码带宽特征 - 使用1D卷积处理时间序列数据
        self.bandwidth_encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),  # 时间维度池化为1
            nn.Flatten()  # 输出形状: [batch_size, embed_dim]
        )
        
        # 编码FOV历史特征 - 使用两层卷积处理
        self.fov_encoder = nn.Sequential(
            nn.Conv1d(in_channels=TILE_IN_F, out_channels=embed_dim//2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=embed_dim//2, out_channels=embed_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),  # 自适应池化
            nn.Flatten()  # 输出形状: [batch_size, embed_dim]
        )
        
        # 编码缓冲区特征 - 使用全连接层
        self.buffer_encoder = nn.Sequential(
            nn.Linear(1 + TILE_IN_F, embed_dim),
            nn.LeakyReLU()
        )
        
        # 编码视频大小特征
        self.gof_size_encoder = nn.Sequential(
            nn.Linear(TILE_IN_F * QUALITY_LEVELS, embed_dim),
            nn.LeakyReLU()
        )
        
        # 编码帧计数器
        self.frame_counter_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LeakyReLU()
        )
        
        # 输出层 - 融合所有特征
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        前向传播函数。

        Args:
            state: 输入状态张量，形状为 [batch_size, seq_len, feature_dim]

        Returns:
            encoded_state: 编码后的状态特征，形状为 [batch_size, embed_dim]
        """
        # 检查输入维度
        if len(state.shape) != 3:
            raise ValueError(f"输入状态必须是3D张量 [batch_size, seq_len, feature_dim]，实际得到 {state.shape}")
        
        batch_size, seq_len, feature_dim = state.shape
        
        # 提取带宽特征，预期在前3个特征维度
        bandwidth_dim = min(3, feature_dim)
        bandwidth_features = state[:, :, :bandwidth_dim]  # [batch_size, seq_len, bandwidth_dim]
        
        # 转置为卷积层期望的格式 [batch_size, channels, seq_len]
        bandwidth_features = bandwidth_features.transpose(1, 2)  # [batch_size, bandwidth_dim, seq_len]
        
        # 确保维度足够，如果不足则填充
        if bandwidth_features.shape[1] < 3:
            padding = torch.zeros(batch_size, 3 - bandwidth_features.shape[1], seq_len, 
                                 device=state.device)
            bandwidth_features = torch.cat([bandwidth_features, padding], dim=1)
        
        # FOV历史特征，预期在bandwidth后面
        fov_start = bandwidth_dim
        fov_length = 50  # 历史FOV长度
        fov_end = min(fov_start + fov_length, feature_dim)
        
        # 创建FOV特征，形状为 [batch_size, TILE_IN_F, fov_length]
        if fov_end > fov_start:
            # 提取可用的FOV特征
            fov_features = state[:, -1, fov_start:fov_end].unsqueeze(1)  # [batch_size, 1, fov_avail]
            fov_features = fov_features.repeat(1, TILE_IN_F, 1)  # 复制到所有Tile
        else:
            # 如果没有FOV特征，创建零张量
            fov_features = torch.zeros(batch_size, TILE_IN_F, fov_length, device=state.device)
        
        # 缓冲区特征，包括缓冲区大小和状态
        buffer_start = fov_end
        buffer_size = state[:, -1, buffer_start:buffer_start+1] if buffer_start < feature_dim else torch.zeros(batch_size, 1, device=state.device)
        
        buffer_state_start = buffer_start + 1
        buffer_state_end = min(buffer_state_start + TILE_IN_F, feature_dim)
        
        if buffer_state_end > buffer_state_start:
            buffer_state = state[:, -1, buffer_state_start:buffer_state_end]
            # 如果缓冲区状态不足TILE_IN_F，则填充
            if buffer_state.shape[1] < TILE_IN_F:
                padding = torch.zeros(batch_size, TILE_IN_F - buffer_state.shape[1], device=state.device)
                buffer_state = torch.cat([buffer_state, padding], dim=1)
        else:
            buffer_state = torch.zeros(batch_size, TILE_IN_F, device=state.device)
        
        buffer_features = torch.cat([buffer_size, buffer_state], dim=1)  # [batch_size, 1+TILE_IN_F]
        
        # GOF大小特征
        gof_start = buffer_state_end
        gof_size_dim = TILE_IN_F * QUALITY_LEVELS
        gof_end = min(gof_start + gof_size_dim, feature_dim)
        
        if gof_end > gof_start:
            gof_size_features = state[:, -1, gof_start:gof_end]
            # 如果GOF大小特征不足预期维度，则填充
            if gof_size_features.shape[1] < gof_size_dim:
                padding = torch.zeros(batch_size, gof_size_dim - gof_size_features.shape[1], device=state.device)
                gof_size_features = torch.cat([gof_size_features, padding], dim=1)
        else:
            gof_size_features = torch.zeros(batch_size, gof_size_dim, device=state.device)
        
        # 帧计数器特征
        frame_counter = state[:, -1, -1:] if feature_dim > 0 else torch.zeros(batch_size, 1, device=state.device)
        
        # 编码各个特征
        encoded_bandwidth = self.bandwidth_encoder(bandwidth_features)  # [batch_size, embed_dim]
        encoded_fov = self.fov_encoder(fov_features)  # [batch_size, embed_dim]
        encoded_buffer = self.buffer_encoder(buffer_features)  # [batch_size, embed_dim]
        encoded_gof_size = self.gof_size_encoder(gof_size_features)  # [batch_size, embed_dim]
        encoded_frame = self.frame_counter_encoder(frame_counter)  # [batch_size, embed_dim]
        
        # 特征融合
        combined_features = torch.cat([
            encoded_bandwidth,
            encoded_fov,
            encoded_buffer,
            encoded_gof_size,
            encoded_frame
        ], dim=1)  # [batch_size, embed_dim*5]
        
        # 输出编码后的状态
        encoded_state = self.output_layer(combined_features)  # [batch_size, embed_dim]
        
        return encoded_state