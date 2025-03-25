
from utils import load_data
from model_pcv.config import cfg
import numpy as np    
all_cooked_bw=[]
all_cooked_time=[]
trace=[0,714, 486, 1196, 352, 1312, 1192, 1062, 97, 82, 847, 1198, 94, 892, 1040, 1349, 1204, 1054, 300, 614, 387, 386, 807, 1297, 909, 1453, 128, 683, 632, 271, 1265, 1226, 1364, 1420, 597, 1288, 132, 245, 744, 209, 1285]
for startpos in range(1,41):
    cooked_time, cooked_bw = load_data.load_trace(filename=cfg.trace_dirs['trace_5g']+'trace_5g.txt',startposition=startpos)
    all_cooked_bw.append(cooked_bw)#Mbps
    all_cooked_time.append(cooked_time)
video_size = load_data.load_video('./data_pcv/cooked_data/tile_counts_longdress.txt')#Mb
cur_gof_size=0
for frame in range(30):
        for tile in range(12):
            # 如果tile可见，则累加视频大小
            # if selected_tile[tile]>0.1:
            cur_gof_size+=video_size[180+frame][tile][0]
print(all_cooked_bw[2][0])
print(cur_gof_size)