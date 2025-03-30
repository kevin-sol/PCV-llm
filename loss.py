import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

loss_path = './loss_log.txt'
# 读取损失日志文件
df = pd.read_csv(loss_path)

# 计算每100个episode的平均值
window_size = 100
grouped_df = df.copy()
grouped_df['Episode_Group'] = (grouped_df['Episode'] // window_size) * window_size  # 对每100个episode分组
avg_df = grouped_df.groupby('Episode_Group').mean().reset_index()

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(df['Episode'], df['Loss'], 'b-', alpha=0.3, label='loss')
plt.plot(avg_df['Episode_Group'], avg_df['Loss'], 'r-', linewidth=2, label=f'avg:{window_size} episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# 可以同时绘制多个指标
plt.figure(figsize=(12, 8))


# 奖励子图
plt.subplot(2, 2, 1)
plt.plot(df['Episode'], df['Reward'], 'g-', alpha=0.3, label='Original value')
plt.plot(avg_df['Episode_Group'], avg_df['Reward'], 'r-', linewidth=2, label=f'avg:{window_size}episode')
plt.title('Reward')
plt.grid(True)
plt.legend()

# 质量子图
plt.subplot(2, 2, 2)
plt.plot(df['Episode'], df['Quality'], 'c-', alpha=0.3, label='Original value')
plt.plot(avg_df['Episode_Group'], avg_df['Quality'], 'r-', linewidth=2, label=f'avg:{window_size}episode')
plt.title('Quality')
plt.grid(True)
plt.legend()

# 重缓冲子图
plt.subplot(2, 2, 3)
plt.plot(df['Episode'], df['Rebuffer'], 'm-', alpha=0.3, label='Original value')
plt.plot(avg_df['Episode_Group'], avg_df['Rebuffer'], 'r-', linewidth=2, label=f'avg:{window_size}episode')
plt.title('Rebuffer')
plt.grid(True)
plt.legend()


# switch子图
plt.subplot(2, 2, 4)
plt.plot(df['Episode'], df['Switch'], 'b-', alpha=0.3, label='Original value')
plt.plot(avg_df['Episode_Group'], avg_df['Loss'], 'r-', linewidth=2, label=f'avg:{window_size}episode')
plt.title('Switch')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('metrics_curves.png')
plt.show()
