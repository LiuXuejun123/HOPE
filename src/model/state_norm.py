"""观测归一化(State Normalization), 一种常见的 RL 训练技巧: 用滑动均值和标准差把观测缩放到相近的数值范围。"""

import numpy as np

DEFAULT_UPDATE_MODAL = {'img':False, 'lidar':True, 'target':True, 'action_mask':False}

class StateNorm():
    """对每个观测模态分别维护滑动均值和标准差(Welford 在线算法)。
    默认只归一化 'lidar' 和 'target', 'img'(已在 [0,1])和 'action_mask'(已在 [0,1])不处理。
    注意: 归一化统计量随模型一起保存, 加载模型时也会一并恢复。
    """
    def __init__(self, observation_shape:dict, update_modal:dict=DEFAULT_UPDATE_MODAL) -> None:
        """参数:
            observation_shape: {模态名: 形状}。
            update_modal:      {模态名: 是否需要归一化}。
        """
        self.observation_shape = observation_shape
        self.update_modal = update_modal
        self.n_state = 0
        self.state_mean, self.S, self.state_std = {}, {}, {}
        for obs_type in self.observation_shape.keys():
            self.state_mean[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)
            self.S[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)
            self.state_std[obs_type] = np.sqrt(self.S[obs_type])
        self.fixed = False

    def fix_parameters(self,):
        """冻结统计量, 之后即使 update=True 也不再更新。"""
        self.fixed = True
    
    def init_state_norm(self, mean, std, S, n_state):
        """手动设置统计量(当前未使用)。"""
        self.n_state = n_state
        self.mean, self.std, self.S = mean, std, S

    def state_norm(self, observation: dict, update=False):
        """归一化一个观测 dict(会原地修改并返回)。

        参数:
            observation: 观测 dict。
            update:      是否用这个观测更新均值/标准差。
        返回:
            归一化后的观测: (x - mean) / (std + 1e-8)。
        """
        if self.n_state == 0:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                if self.update_modal[obs_type]:
                    self.state_mean[obs_type] = observation[obs_type]
                    self.state_std[obs_type] = observation[obs_type]
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        elif update==False or self.fixed:
            for obs_type in self.observation_shape.keys():
                if self.update_modal[obs_type]:
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        elif update==True:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                if self.update_modal[obs_type]:
                    old_mean = self.state_mean[obs_type].copy()
                    self.state_mean[obs_type] = old_mean + (observation[obs_type] - old_mean) / self.n_state
                    self.S[obs_type] = self.S[obs_type] + (observation[obs_type] - old_mean) *\
                        (observation[obs_type] - self.state_mean[obs_type])
                    self.state_std[obs_type] = np.sqrt(self.S[obs_type] / self.n_state)
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        return observation
