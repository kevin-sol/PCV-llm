import numpy as np
from model_pcv.Hyperparameters import VIDEO_GOF_LEN,F_IN_GOF,TILE_IN_F,\
    PACKET_PAYLOAD_PORTION,DECODING_TIME_RATIO,FRAME
from utils.logger import setup_logger
#RANDOM_SEED = Hyperparameters.RANDOM_SEED
# #每个gof的时间!!!!!!!!!!!!!!!!!!
# VIDEO_GOF_LEN = Hyperparameters.VIDEO_GOF_LEN #秒
# #每个GOF有N个'F' 30
# F_IN_GOF=Hyperparameters.F_IN_GOF
# #一个点云切块2*3*2
# TILE_IN_F=Hyperparameters.TILE_IN_F
# PACKET_PAYLOAD_PORTION =Hyperparameters.PACKET_PAYLOAD_PORTION
# DECODING_TIME_RATIO=Hyperparameters.DECODING_TIME_RATIO
# FRAME=Hyperparameters.FRAME
# BUFFER_THRESH = 2  # 缓冲区阈值（秒）
# DRAIN_BUFFER_SLEEP_TIME =0.5  # 缓冲区排空睡眠时间（秒）

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw,video_size,random_seed):
        
        np.random.seed(random_seed)
        self.logger = setup_logger()
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.cooked_time = all_cooked_time[0]
        self.cooked_bw = all_cooked_bw[0]
        self.video_frame_counter = 0
        self.buffer_size = 0
        #这些指针用于遍历预先定义的网络条件数据（存储在 cooked_time 和 cooked_bw 中）。
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1
        #记录了上一个网络条件更新的时间点。 
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_size=video_size
        self.buffer=[]
        #初始化缓冲区，缓冲区的大小为视频帧数除以每个GOF的帧数，每个GOF的缓冲区大小为tile的数量。
        for i in range(FRAME//F_IN_GOF):
            self.buffer.append([])
            for j in range(TILE_IN_F):
                self.buffer[i].append(-1)        
    def reset(self):
        """重置环境到初始状态"""
        # 选择随机网络轨迹
        trace_idx = np.random.randint(len(self.all_cooked_bw))
        #self.logger.info(f"选择网络轨迹 {trace_idx}")
        #self.logger.info(f"网络轨迹统计: 平均带宽={np.mean(self.all_cooked_bw[trace_idx])}, 最大={np.max(self.all_cooked_bw[trace_idx])}, 最小={np.min(self.all_cooked_bw[trace_idx])}")
    
        self.cooked_time = self.all_cooked_time[trace_idx]
        self.cooked_bw = self.all_cooked_bw[trace_idx]
        
        # 网络指针
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        
        # 播放状态
        self.video_frame_counter = 0  # 当前帧索引
        self.buffer_size = 0.0  # 缓冲区大小(s)
        
        # 初始化缓冲区，-1表示未下载
        self.buffer = [[-1] * TILE_IN_F for _ in range(len(self.video_size) // F_IN_GOF + 1)]
        
        # 统计信息
        self.total_rebuffer = 0.0
        self.total_delay = 0.0
        self.total_gof_size = 0
        
        # 播放状态
        #self.played_time = 0.0
        #self.current_time = 0.0
        
        return True
                
    # 计算下载一个视频GOF所需的时间，并更新缓冲区
    def get_video_gof(self, selected_tile,selected_quality):
        # 记录起始状态
        #self.logger.info(f"--- 开始下载新 GOF ---")
        #self.logger.info(f"视频帧计数器: {self.video_frame_counter}")
        #self.logger.info(f"初始缓冲区大小: {self.buffer_size} 秒")
        #self.logger.info(f"当前带宽: {self.cooked_bw[self.mahimahi_ptr]} Mbps")
        
        # 记录选择的 tile 和质量
        #tile_log = [i for i, val in enumerate(selected_tile) if val > 0.1]
        #self.logger.info(f"选择的 tile: {tile_log}")
        #quality_log = [selected_quality[i] for i in tile_log]
        #self.logger.info(f"对应质量: {quality_log}")
        
        delay = 0.0  
        sleep_time = 0.0
        rebuffer = 0.0
        # 初始化下载计数器
        video_gof_counter_sent = 0  
        # 初始化当前GOF的大小
        cur_gof_size=0
        #遍历当前GOF的每个帧，并计算gof大小
        for frame in range(F_IN_GOF):
            for tile in range(TILE_IN_F):
                # 如果tile可见，则累加视频大小
                if selected_tile[tile]>0.1:
                    cur_gof_size+=self.video_size[self.video_frame_counter+frame][tile][selected_quality[tile]]
                    
        #print(f"当前帧：{self.video_frame_counter},gof_size:{cur_gof_size}") 
        # 加上解码时间
        #self.logger.info(f"cur_gof_size: {cur_gof_size}")
        delay+=cur_gof_size*DECODING_TIME_RATIO#decoding time
        # 遍历网络条件数据，模拟下载视频GOF的过程
        while True:  # download video chunk over mahimahi
            # 获取当前网络的吞吐量
            throughput = self.cooked_bw[self.mahimahi_ptr]
            #print(f"指针：{self.mahimahi_ptr},带宽：{throughput}")
            # 计算当前网络吞吐量下的下载时间
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
            # 计算当前网络吞吐量下的下载数据量
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            # 如果当前网络吞吐量下的下载大小超过了当前GOF的大小，则计算剩余时间并退出循环
            if video_gof_counter_sent + packet_payload > cur_gof_size:
                # 
                fractional_time=(cur_gof_size-video_gof_counter_sent)/throughput/PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            # 更新下载计数器和延迟
            video_gof_counter_sent += packet_payload
            delay += duration
            # 更新上一个网络条件更新的时间点
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            # 移动到下一个网络条件数据
            self.mahimahi_ptr += 1

            # 如果网络条件数据遍历完毕，则循环回到开始
            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
                pass
        
        #self.logger.info(f"计算的延迟: {delay}")
        # 计算 rebuffer 之前记录
        #self.logger.info(f"计算 rebuffer 前缓冲区: {self.buffer_size}")
        rebuffer = max(delay - self.buffer_size, 0.0)
        #print(f"rebuffer:{rebuffer}")
        #self.logger.info(f"计算的 rebuffer: {rebuffer}")
        
        # 计算重新缓冲时间
        #rebuffer = max(delay - self.buffer_size, 0.0)#秒
        
        # 更新缓冲区
        self.buffer_size = max(self.buffer_size - delay, 0.0)
        #print(f"减去延迟后缓冲区:{self.buffer_size}")
        #self.logger.info(f"减去延迟后缓冲区: {self.buffer_size}")

        
        #如果缓冲区过大，就进行睡眠
        # if self.buffer_size > BUFFER_THRESH:
        #     drain_buffer_time = self.buffer_size - BUFFER_THRESH
        #     sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME
        #     self.buffer_size -= sleep_time
            
        #     # 处理睡眠时间
        #     while True:
        #         duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
        #         if duration > sleep_time :
        #             self.last_mahimahi_time += sleep_time 
        #             break
        #         sleep_time -= duration *
        #         self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
        #         self.mahimahi_ptr += 1

        #         # 如果指针超出范围，则循环回到开始位置
        #         if self.mahimahi_ptr >= len(self.cooked_bw):
        #             self.mahimahi_ptr = 1
        #             self.last_mahimahi_time = 0
        
        # 更新缓冲区
        for tile in range(TILE_IN_F):
            if selected_tile[tile]>0.1:
                self.buffer[int(self.video_frame_counter/F_IN_GOF)][tile]=selected_quality[tile]
        
        self.buffer_size += VIDEO_GOF_LEN
        #self.logger.info(f"添加 GOF 长度后缓冲区: {self.buffer_size}")

        self.video_frame_counter += F_IN_GOF
        # 判断是否到达视频末尾
        end_of_video = False
        if self.video_frame_counter>= len(self.video_size)-1:
            end_of_video = True
        # 计算剩余GOF数量
        gof_remain = (len(self.video_size) - self.video_frame_counter) // F_IN_GOF
    
        return delay, sleep_time, self.buffer_size, rebuffer, cur_gof_size, end_of_video, gof_remain, self.buffer    
    