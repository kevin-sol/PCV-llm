import pandas as pd
import matplotlib.pyplot as plt

loss_path='./loss_log.txt'
# 读取损失日志文件
df = pd.read_csv(loss_path)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(df['Episode'], df['Loss'], label='Training Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# 可以同时绘制多个指标
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(df['Episode'], df['Loss'])
plt.title('Loss')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(df['Episode'], df['Reward'])
plt.title('Reward')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(df['Episode'], df['Quality'])
plt.title('Quality')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(df['Episode'], df['Rebuffer'])
plt.title('Rebuffer')
plt.grid(True)

plt.tight_layout()
plt.savefig('metrics_curves.png')
plt.show()