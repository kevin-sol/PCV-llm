import numpy as np
from model_pcv.Hyperparameters import VIDEO_GOF_LEN,F_IN_GOF,TILE_IN_F,\
    PACKET_PAYLOAD_PORTION,DECODING_TIME_RATIO,FRAME
from utils.logger import setup_logger

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw,video_size,random_seed):
        
        np.random.seed(random_seed)
        self.logger = setup_logger()
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.cooked_time = all_cooked_time[0]
        self.cooked_bw = all_cooked_bw[0]
        
        # 下载帧计数器 - 表示已下载的帧索引
        self.video_frame_counter = 0
        
        # 播放位置计数器 - 新增：表示当前正在播放的帧索引
        self.playback_position = 0
        
        self.buffer_size = 0
        #这些指针用于遍历预先定义的网络条件数据（存储在 cooked_time 和 cooked_bw 中）。
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1
        #记录了上一个网络条件更新的时间点。 
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_size=video_size
        self.buffer=[]
        
        # 累积播放时间 - 新增：用于跟踪模拟的播放时间
        self.accumulated_playback_time = 0.0
        
        #初始化缓冲区，缓冲区的大小为视频帧数除以每个GOF的帧数，每个GOF的缓冲区大小为tile的数量。
        for i in range(FRAME//F_IN_GOF):
            self.buffer.append([])
            for j in range(TILE_IN_F):
                self.buffer[i].append(-1)        
          
    def reset(self):
        """重置环境到初始状态"""
        # 选择随机网络轨迹
        trace_idx = np.random.randint(len(self.all_cooked_bw))
        
        self.cooked_time = self.all_cooked_time[trace_idx]
        self.cooked_bw = self.all_cooked_bw[trace_idx]
        
        # 网络指针
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        
        # 下载和播放状态
        self.video_frame_counter = 0  # 当前下载帧索引
        self.playback_position = 0    # 当前播放帧索引 - 新增
        self.buffer_size = 0.0  # 缓冲区大小(s)
        self.accumulated_playback_time = 0.0  # 累积播放时间 - 新增
        
        # 初始化缓冲区，-1表示未下载
        self.buffer = [[-1] * TILE_IN_F for _ in range(len(self.video_size) // F_IN_GOF + 1)]
        
        # 统计信息
        self.total_rebuffer = 0.0
        self.total_delay = 0.0
        self.total_gof_size = 0
        
        return True
                
    # 计算下载一个视频GOF所需的时间，并更新缓冲区
    def get_video_gof(self, selected_tile, selected_quality):
        # 原有的下载逻辑...
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
        
        # 加上解码时间
        delay+=cur_gof_size*DECODING_TIME_RATIO
        
        # 原有的带宽模拟逻辑...
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr]
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_gof_counter_sent + packet_payload > cur_gof_size:
                fractional_time=(cur_gof_size-video_gof_counter_sent)/throughput/PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_gof_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
                pass
        
        # 计算重新缓冲时间
        rebuffer = max(delay - self.buffer_size, 0.0)
        
        # 更新缓冲区
        self.buffer_size = max(self.buffer_size - delay, 0.0)
        
        # 更新缓冲区内容
        for tile in range(TILE_IN_F):
            if selected_tile[tile]>0.1:
                self.buffer[int(self.video_frame_counter/F_IN_GOF)][tile]=selected_quality[tile]
        
        self.buffer_size += VIDEO_GOF_LEN
       
        # 更新播放位置 - 新增
        # 如果发生了rebuffer，播放位置不变，否则正常播放
        if rebuffer > 0:
            # 重缓冲情况下播放位置不变，等待缓冲区增加
            self.playback_position=self.video_frame_counter
        else:
            # 从当前播放位置播放，直到下一个下载开始或播放到当前下载的内容末尾
            # 播放位置不能超过已下载内容的位置
            self.playback_position+=delay*F_IN_GOF/VIDEO_GOF_LEN
           
        # 更新下载位置
        self.video_frame_counter += F_IN_GOF
         
        # 判断是否到达视频末尾
        end_of_video = False
        if self.video_frame_counter >= len(self.video_size)-1:
            end_of_video = True
            
        # 计算剩余GOF数量
        gof_remain = (len(self.video_size) - self.video_frame_counter) // F_IN_GOF
    
        return delay, sleep_time, self.buffer_size, rebuffer, cur_gof_size, end_of_video, gof_remain, self.buffer
        
    # 新增方法：获取当前播放位置
    def get_playback_position(self):
        """
        获取当前播放位置
        
        Returns:
            当前正在播放的帧索引
        """
        return self.playback_position