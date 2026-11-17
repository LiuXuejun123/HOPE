"""环境包装器: 把原始 CarParking 包装成标准的 RL 接口。

    - 动作: 网络输出的 [-1, 1] 线性映射到真实转角/速度范围;
    - 奖励: 把奖励分量加权成一个标量, 并在终止时给出大的奖惩;
    - 观测: 图像从 HWC 转成 PyTorch 习惯的 CHW;
    - step 返回 (obs, reward, done, info) 四元组。
"""


import numpy as np
from gym import Wrapper

from env.car_parking_base import CarParking
from env.vehicle import Status
from configs import REWARD_WEIGHT, REWARD_RATIO

def reward_shaping(*args):
    """把原始环境返回的奖励分量合成为标量奖励。

    参数:
        args: 原始环境 step() 的返回值 (obs, reward_info, status, info)。
    返回:
        (obs, reward, status, info), 其中 info['status'] 记录了状态。

    奖励规则:
        CONTINUE: sum(REWARD_WEIGHT[k] * reward_info[k]);
        ARRIVED: +50;  COLLIDED / OUTBOUND: -50;  OUTTIME: -1;
        最后统一乘以 REWARD_RATIO(0.1)。
    """
    obs, reward_info, status, info = args
    if status == Status.CONTINUE:
        reward = 0
        for reward_type in REWARD_WEIGHT.keys():
            reward += REWARD_WEIGHT[reward_type]*reward_info[reward_type]
    elif status == Status.OUTBOUND:
        reward = -50
    elif status == Status.OUTTIME:
        reward = -1
    elif status == Status.ARRIVED:
        reward = 50
    elif status == Status.COLLIDED:
        reward = -50
    else:
        print(status)
        print('Never reach here !!!')
    reward *= REWARD_RATIO
    info['status'] = status
    return obs, reward, status, info

def action_rescale(action:np.ndarray, action_space, raw_action_range=(-1,1), explore:bool=True, epsilon:float=0.0):
    """把网络输出的动作从 raw_action_range(默认 [-1, 1]) 线性映射到环境的真实动作范围。

    参数:
        action:       网络输出的动作。
        action_space: 原始环境的动作空间(Box)。
        explore:      是否允许 epsilon-greedy 随机探索。
        epsilon:      随机探索的概率(默认 0, 即不探索)。
    返回:
        真实物理量的动作 [steer, speed]。
    """
    action = np.clip(action, *raw_action_range)
    action = action * (action_space.high - action_space.low) / 2 + (action_space.high + action_space.low) / 2
    if explore and np.random.random() < epsilon:
        action = action_space.sample()
    return action

def observation_rescale(obs):
    """把图像观测从 (H, W, C) 转为 (C, H, W)。"""
    if obs['img'] is not None:
        obs['img'] = obs['img'].transpose((2,0,1))
    return obs


class CarParkingWrapper(Wrapper):
    """CarParking 的 gym Wrapper, 训练和评估脚本实际交互的环境对象。"""
    def __init__(self, env:CarParking, 
            action_func:callable=action_rescale, 
            reward_func:callable=reward_shaping,
            observation_func:callable=observation_rescale,
            ):
        """参数:
            env: 原始 CarParking 环境。
            action_func / reward_func / observation_func: 动作、奖励、观测的变换函数, 可替换。
        同时记录各观测模态的形状 observation_shape(图像为 CHW), 供网络构建使用。
        """
        super().__init__(env)
        self.reward_func = reward_func
        self.action_func = action_func
        self.obs_func = observation_func
        self.observation_shape = {k:self.env.observation_space[k].shape for k in self.env.observation_space}
        if 'img' in self.observation_shape:
            w,h,c = self.observation_shape['img']
            self.observation_shape['img'] = (c,w,h)

    def step(self, action=None):
        """执行一步。

        参数:
            action: 网络输出的 [-1, 1] 动作; 为 None 时只返回当前观测。
        返回:
            obs (dict), reward (float), done (bool, 非 CONTINUE 即为 True), info (含 status、reward_info、path_to_dest)。
        """
        if action is None:
            return self.obs_func(self.env.step()[0])
        action = self.action_func(action, self.env.action_space)
        returns = self.env.step(action)
        obs, reward, status, info = self.reward_func(*returns)
        obs = self.obs_func(obs)
        done = False if status==Status.CONTINUE else True
        return obs, reward, done, info

    def reset(self, *args):
        """重置环境, 参数透传给 CarParking.reset(case_id, data_dir, level), 返回初始观测。"""
        obs = self.env.reset(*args)
        return self.obs_func(obs)

