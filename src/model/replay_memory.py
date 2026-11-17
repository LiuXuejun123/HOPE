"""经验缓存。PPO 用它当作"一批 on-policy 数据"的缓冲区, SAC 用它当作经验回放池。"""
from collections import deque

import numpy as np


class ReplayMemory(object):
    """按字段分别存储的经验缓冲区, 每个字段是一个有最大长度的 deque(满了会自动丢弃最旧的数据)。
    默认字段为 state、action、reward、done, 可通过 extra_items 增加(本项目加了 log_prob、next_obs)。
    """
    def __init__(self, memory_size: int, extra_items: list = []):
        """参数:
            memory_size: 最大容量。
            extra_items: 额外字段名列表。
        """
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.memory = {}
        for item in self.items:
            self.memory[item] = deque([], maxlen=memory_size)
    
    def push(self, observations:tuple):
        """存入一条经验, observations 是按字段顺序排列的元组。"""
        for i, item in enumerate(self.items):
            self.memory[item].append(observations[i])

    def get_items(self, idx_list: np.ndarray):
        """按下标取出一批经验。

        返回:
            dict, 每个字段对应一个列表(元素为 ndarray 时会转成 ndarray);
            额外包含 "next_state": 下一条的 state, episode 结束或已是最后一条时为 None。
        """
        batches = {}
        for item in self.items:
            batches[item] = []
        batches["next_state"] = []
        for idx in idx_list:
            for item in self.items:
                batches[item].append(self.memory[item][idx])
            if idx == self.__len__()-1 or self.memory["done"][idx]:
                batches["next_state"].append(None)
            else:
                batches["next_state"].append(self.memory["state"][idx+1])
        for idx in batches.keys():
            if isinstance(batches[idx][0], np.ndarray):
                batches[idx] = np.array(batches[idx])
        return batches

    def sample(self, batch_size: int):
        """随机(有放回)采样 batch_size 条经验, SAC 使用。"""
        idx_list = np.random.randint(self.__len__(), size=batch_size)
        return self.get_items(idx_list)

    def shuffle(self, idx_range: int = None):
        """打乱顺序取出前 idx_range 条经验(当前未使用)。"""
        idx_range = self.__len__() if idx_range is None else idx_range
        idx_list = np.arange(idx_range)
        np.random.shuffle(idx_list)
        return self.get_items(idx_list)

    def clear(self):
        """清空所有经验, PPO 每次更新后调用。"""
        for item in self.items:
            self.memory[item].clear()

    def __len__(self):
        """当前存储的经验条数。"""
        return len(self.memory["state"])
