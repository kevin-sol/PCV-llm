import os
from model_pcv import Hyperparameters
import torch
import torch.nn as nn
import numpy as np

from model_pcv.controller import LlamaPointCloudController
from model_pcv.rl_policy import PointCloudRLPolicy
from model_pcv.state_encoder import EncoderNetwork
from model_pcv.env_pcv import Environment
from utils.plm_utils import load_plm
from utils.plm_utils import set_random_seed
from model_pcv.config import cfg
from utils import load_data
from utils.model import save_model, load_model
from model_pcv.Hyperparameters import TILE_IN_F,F_IN_GOF,QUALITY_LEVELS,VIDEO_GOF_LEN,\
    FRAME,INIT_QOE,REBUF_PENALTY,SMOOTH_PENALTY,MULTIPLE_QUALITY_LEVELS
from utils.logger import setup_logger
logger=setup_logger("main_log")
# 定义不同预训练语言模型的层数配置
PLM_LAYER_SIZES = {
    'gpt2': {
        'base': 24,
        'small': 12,
        'large': 36,
        'xl': 48
    },
    'llama': {
        'base': 32,
    },
    't5-lm': { 
        'base': 12,
        'small': 6,
        'large': 24,
        'xl': 24
    }
}


def main():
    """主函数 - 训练基于Llama-2-7B的点云流媒体系统"""
    # 解析参数
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir_test', type=str, default=cfg.plm_ft_dir+'best_model')

    # 数据参数
    #parser.add_argument('--trace_dir', type=str, default='./traces')
    parser.add_argument('--video_size_file', type=str, default='./data_pcv/cooked_data/tile_counts_longdress.txt')
    #parser.add_argument('--fov_traces_file', type=str, default='./fov_traces.npy')
    #parser.add_argument('--save_dir', type=str, default='./saved_models')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--residual', action='store_true', help='使用残差连接')
    parser.add_argument('--seed', help='random seed', type=int, default=100003)
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--plm_embed_size',help='设置为与你加载的预训练模型本身的隐藏层大小一致', type=int, default=4096)
    
    parser.add_argument('--report_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--max_length', type=int, help='用于控制历史记忆的长度',default=50)
    parser.add_argument('--max_ep_len', type=int, default=10)
    parser.add_argument('--which_layer', type=int, default=-1, help='提前停止的层')
    
    # 语言模型参数.
    parser.add_argument('--plm_dir', type=str, default=cfg.plm_dir)
    parser.add_argument('--plm_type', type=str, default='llama')
    parser.add_argument('--plm_size', type=str, default='base')
    parser.add_argument('--rank', type=int, default=128, help='LoRA低秩适应的秩')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--device_out', type=str, default='cuda:1')  # 输出设备 
    parser.add_argument('--device_mid', type=str, default=None)  # 中间设备 
    parser.add_argument('--train', action='store_true', help='是否进行训练')
    parser.add_argument('--test', action='store_true', help='是否进行测试')
    
    # state encoder settings
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder', default=256)
    
    args = parser.parse_args()
    
    # 创建保存目录
    
    #os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 加载数据
    all_cooked_bw=[]
    all_cooked_time=[]
    trace=[0,714, 486, 1196, 352, 1312, 1192, 1062, 97, 82, 847, 1198, 94, 892, 1040, 1349, 1204, 1054, 300, 614, 387, 386, 807, 1297, 909, 1453, 128, 683, 632, 271, 1265, 1226, 1364, 1420, 597, 1288, 132, 245, 744, 209, 1285]
    for startpos in range(1,41):
        cooked_time, cooked_bw = load_data.load_trace(filename=cfg.trace_dirs['trace_5g']+'trace_5g.txt',startposition=startpos)
        all_cooked_bw.append(cooked_bw)#Mbps
        all_cooked_time.append(cooked_time)
    #print(len(all_cooked_bw))
    # for i in range(5):
    #     cooked_time, cooked_bw = load_data.load_trace(cfg.trace_dirs['trace_5g']+str(i+1)+'.txt')
    #     all_cooked_bw.append(cooked_bw)#Mbps
    #     all_cooked_time.append(cooked_time)
    print("带宽数据加载完成...")
    video_size = load_data.load_video(args.video_size_file)#Mb
    print("视频大小数据加载完成...")
    fov_traces = []
    fov_path = cfg.trace_dirs['vp']+'longdress/'
    for i in range(1,31):
        fov_traces.append(load_data.load_fov(fov_path+'p'+str(i)+'_fovs_longdress'+'.txt'))
    print("训练fov数据加载完成...")
    dis_traces = []
    dis_path = cfg.trace_dirs['vp']+'longdress/'
    for i in range(1,31):
        dis_traces.append(load_data.load_dis(dis_path+'p'+str(i)+'_dis_longdress'+'.txt'))
    print("训练dis数据加载完成...")
    # 创建环境
    
    env = Environment(
        all_cooked_time=all_cooked_time,  # 网络轨迹时间数据 
        all_cooked_bw=all_cooked_bw,      # 网络轨迹带宽数据 
        video_size=video_size,    
        random_seed=args.seed                              
    )
    
    # 创建点云流媒体控制器
    controller = LlamaPointCloudController(
        args=args,
        env=env,
        video_size=video_size,
        device=args.device
    )
    # 训练
    if args.train:
        print("开始训练...")
        stats = controller.train(
            fov_traces=fov_traces, 
            dis_traces=dis_traces,
            num_episodes=args.num_episodes,
            batch_size=args.batch_size,
            report_interval=args.report_interval,
            eval_interval=args.eval_interval, 
            save_interval=args.eval_interval, 
            model_dir=cfg.plm_ft_dir
        )
        
        model_save_dir = cfg.plm_ft_dir
        #final_model_save_dir = os.path.join(model_save_dir, "final_model")
        # 保存训练结果
        #np.save(os.path.join(model_save_dir, "training_stats.npy"), stats)
        #保存模型
        #save_model(args, controller.policy, final_model_save_dir)
        #torch.save(controller.plm.state_dict(), os.path.join(final_model_save_dir, 'pc_llama_adapted.pth'))
        # torch.save(controller.state_encoder.state_dict(), os.path.join(final_model_save_dir, 'pc_state_encoder.pth'))
        # torch.save(controller.policy.modules_except_plm.state_dict(), os.path.join(final_model_save_dir, 'policy_modules.pth'))
                
        print(f"训练完成，结果保存至 {model_save_dir}")
        #print(f"模型保存至 {final_model_save_dir}")

    # 测试
    if args.test:
        print("开始测试...")
        
        model_dir_test = args.model_dir_test
        results_dir = cfg.results_dir
        # # 如果存在保存的模型，加载它
        # policy_path = os.path.join(model_dir_test, "policy_modules.pth")
        # if os.path.exists(policy_path):
        #     controller.policy.modules_except_plm.load_state_dict(
        #         torch.load(policy_path, map_location=args.device)
        #     )
        #     print(f"加载保存的策略网络: {policy_path}")
        
        # # 如果使用了低秩适应且存在保存的PLM
        # if args.rank != -1:
        #     plm_path = os.path.join(model_dir_test, "pc_llama_adapted.pth")
        #     if os.path.exists(plm_path):
        #         controller.plm.load_state_dict(
        #             torch.load(plm_path, map_location=args.device)
        #         )
        #         print(f"加载保存的PLM-LoRA: {plm_path}")
        controller.policy=load_model(args, controller.policy, model_dir_test)
        # 记录测试结果
        test_results = {
            'qualities': [],
            'rebuffers': [],
            'switch': []
        }
        
        # 测试每个FOV轨迹
        fov_traces_test = []
        dis_traces_test=[]
        fov_path = cfg.trace_dirs['vp']+'longdress/'
        
        for i in range(31,41):
            fov_traces_test.append(load_data.load_fov(fov_path+'p'+str(i)+'_fovs_longdress'+'.txt'))
            dis_traces_test.append(load_data.load_dis(fov_path+'p'+str(i)+'_dis_longdress'+'.txt'))
        print("fov,dis数据加载完成...")
        
        for i in range(len(fov_traces_test)):
            fov_trace=fov_traces_test[i]
            dis_trace=dis_traces_test[i]
        
            print(f"测试FOV轨迹 {i+1}/{len(fov_traces_test)}...")
            
            # 重置环境和控制器状态
            env.reset()
            controller.fov_history = []
            controller.policy.clear_dq()
            
            # 轨迹的统计数据
            trace_qualities = []
            trace_rebuffers = 0
            trace_switch = []
            
            # 模拟视频播放
            current_frame = 0
            
            prev_quality=[0]*TILE_IN_F
            
            while current_frame < len(env.video_size):
                # 流式传输下一个GOF
                delay , rebuffer , quality, switch,selected_tile, selected_quality ,buffer= controller.stream_next_gof(
                    fov_trace, 
                    dis_trace,
                    current_frame
                )
                # switch = 0.0
                # if(current_frame>0):
                #     for s in range(TILE_IN_F):
                #         # 如果该 tile 被选择（例如 selected_tile[s] > 0.1 表示该 tile 有效）
                #         switch+= \
                #             np.abs(MULTIPLE_QUALITY_LEVELS[max(buffer[current_frame//F_IN_GOF][s],0)]-MULTIPLE_QUALITY_LEVELS[prev_quality[s]])
                #         # switch+= \
                #         #     np.abs(MULTIPLE_QUALITY_LEVELS[max(buffer[current_frame//F_IN_GOF][s],0)]-MULTIPLE_QUALITY_LEVELS[max(buffer[(current_frame//F_IN_GOF-1)][s],0)])
                #         #logger.info(f"switch{s+1}:{switch}")
            
                
                # 记录统计数据
                trace_qualities.append(quality)
                trace_rebuffers += rebuffer
                trace_switch.append(switch)
                
                # 更新当前帧
                current_frame += Hyperparameters.F_IN_GOF
                #更新prev_quality
                prev_quality=selected_quality
                # 如果视频结束，跳出循环
                if current_frame >= len(env.video_size):
                    break
            
            # 计算总质量和rebuffer，switch
            total_quality = np.sum(trace_qualities) 
            total_rebuffer = trace_rebuffers
            total_switch = np.sum(trace_switch) 
            QoE = total_quality*INIT_QOE - total_rebuffer*REBUF_PENALTY -SMOOTH_PENALTY*total_switch
            
            # 添加到测试结果
            test_results['qualities'].append(total_quality)
            test_results['rebuffers'].append(total_rebuffer)
            test_results['switch'].append(total_switch)
            
            print(f"轨迹 {i+1} 结果: QoE={QoE:.0f},质量={total_quality:.4f}, "
                  f"重缓冲={total_rebuffer:.2f}s, switch={total_switch:.4f}")
        
        # 计算总体平均值
        overall_avg_quality = np.mean(test_results['qualities'])
        overall_avg_rebuffer = np.mean(test_results['rebuffers'])
        overall_avg_delay = np.mean(test_results['switch'])
        
        print("\n测试总结:")
        print(f"平均质量: {overall_avg_quality:.4f}")
        print(f"平均重缓冲: {overall_avg_rebuffer:.2f}s")
        print(f"平均质量切换: {overall_avg_delay:.4f}")
        
        # 保存测试结果
        np.save(os.path.join(results_dir, "test_results.npy"), test_results)
        
        # 保存总体平均值到文本文件
        with open(os.path.join(results_dir, "test_summary.txt"), 'w') as f:
            f.write(f"Average Quality: {overall_avg_quality:.4f}\n")
            f.write(f"Average Rebuffer: {overall_avg_rebuffer:.2f}s\n")
            f.write(f"Average Delay: {overall_avg_delay:.4f}\n")
        
        print(f"测试完成，结果保存至 {results_dir}")

if __name__ == '__main__':
    main()