import numpy as np

class PrioritizedBuffer:
    """
    优先级经验回放缓冲区
    使用二叉树结构高效地实现加权采样
    """
    def __init__(self, capacity, alpha=0.6):
        """
        初始化优先级经验回放缓冲区
        
        参数:
            capacity: 缓冲区容量
            alpha: 优先级指数参数 (0 - 无优先级, 1 - 完全按优先级)
        """
        self.capacity = capacity
        self.alpha = alpha
        
        # 存储经验样本的列表
        self.buffer = []
        
        # 存储样本优先级的树结构
        self.priorities = np.zeros(2 * capacity - 1)
        
        # 当前位置和大小
        self.position = 0
        self.size = 0
        
        # 记录最大优先级，用于新样本
        self.max_priority = 1.0
        
    def add(self, experience, priority=None):
        """
        添加新的经验样本到缓冲区
        
        参数:
            experience: 经验样本
            priority: 样本的初始优先级，如果为None则使用最大优先级
        """
        # 如果未指定优先级，则使用最大优先级
        max_priority = self.max_priority if self.size > 0 else 1.0
        priority = priority if priority is not None else max_priority
        
        # 计算叶节点索引
        idx = self.position + self.capacity - 1
        
        # 添加样本
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            # 缓冲区已满，覆盖最老的样本
            self.buffer[self.position] = experience
        
        # 更新优先级
        self.update_priority(idx, priority)
        
        # 更新位置
        self.position = (self.position + 1) % self.capacity
        
    def update_priority(self, idx, priority):
        """
        更新样本的优先级
        
        参数:
            idx: 样本在树中的索引
            priority: 新优先级值
        """
        # 计算优先级变化
        change = priority - self.priorities[idx]
        
        # 更新当前节点
        self.priorities[idx] = priority
        
        # 更新最大优先级记录
        if priority > self.max_priority:
            self.max_priority = priority
        
        # 更新父节点
        parent = (idx - 1) // 2
        while parent >= 0:
            self.priorities[parent] += change
            parent = (parent - 1) // 2
            
    def get_priority(self, idx):
        """
        获取样本的优先级
        
        参数:
            idx: 样本在缓冲区中的索引
            
        返回:
            样本的优先级
        """
        # 计算叶节点索引
        idx = idx + self.capacity - 1
        return self.priorities[idx]
            
    def sample(self, batch_size, beta=0.4):
        """
        按优先级采样
        
        参数:
            batch_size: 批次大小
            beta: 重要性采样指数 (0 - 无重要性采样, 1 - 完全补偿)
            
        返回:
            samples: 采样的经验
            indices: 样本的索引
            weights: 重要性采样权重
        """
        if self.size < batch_size:
            batch_size = self.size  # 采样数量不能超过缓冲区大小
        
        # 存储采样结果
        samples = []
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # 计算优先级总和
        total_priority = self.priorities[0]
        if total_priority == 0:
            # 如果所有优先级都为0，则使用均匀采样
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = np.ones_like(indices, dtype=np.float32)
            samples = [self.buffer[i] for i in indices]
            return samples, indices, weights
        
        # 执行加权采样
        segment = total_priority / batch_size
        
        for i in range(batch_size):
            # 在当前段内随机采样
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # 从树中检索样本
            idx = 0
            while idx < self.capacity - 1:
                left = 2 * idx + 1
                right = left + 1
                
                if left >= len(self.priorities):
                    break
                
                if value <= self.priorities[left]:
                    idx = left
                else:
                    value -= self.priorities[left]
                    idx = right
            
            # 转换为缓冲区索引
            buffer_idx = idx - (self.capacity - 1)
            
            # 确保索引在有效范围内
            if 0 <= buffer_idx < self.size:
                indices[i] = buffer_idx
                # 计算权重
                priority = self.priorities[idx] ** self.alpha
                prob = priority / total_priority
                weight = (self.size * prob) ** (-beta)
                weights[i] = weight
                samples.append(self.buffer[buffer_idx])
        
        # 归一化权重
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        批量更新样本优先级
        
        参数:
            indices: 样本索引列表
            priorities: 对应的新优先级
        """
        for idx, priority in zip(indices, priorities):
            # 确保索引在有效范围内
            if 0 <= idx < self.size:
                # 计算叶节点索引
                tree_idx = idx + self.capacity - 1
                # 更新优先级
                self.update_priority(tree_idx, priority ** self.alpha)