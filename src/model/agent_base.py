"""智能体基类和配置基类, PPOAgent / SACAgent 都继承自这里。"""
from abc import ABC, abstractmethod
import random
from typing import Any

import numpy as np
import torch

from configs import *


class ConfigBase:
    """超参数配置基类。默认值来自 configs.py(GAMMA、BATCH_SIZE、LR、TAU 等),
    子类再添加算法特有的超参数, 最后用 merge_configs 覆盖成调用方传入的值。
    """
    def __init__(self):
        # runtime
        """设置通用默认超参数。"""
        self.n_epoch = None
        self.n_initial_exploration_steps = None

        # environment
        self.state_dim: tuple = None
        self.action_dim: tuple = None

        # model
        self.gamma: float = GAMMA
        self.batch_size = BATCH_SIZE
        self.lr: float = LR
        self.explore: bool = True
        self.explore_config: dict = {}
        self.tau: float = TAU # when tau=0, the update becomes hard update
        self.max_train_steps = MAX_TRAIN_STEP

        # tricks
        self.orthogonal_init = ORTHOGONAL_INIT
        self.lr_decay = LR_DECAY

        # evaluation
        self.evaluation_interval = None

        # save and load
        self.check_list = []
    
    def merge_configs(self, configs: dict):
        """把 configs dict 中的每个键值设为本对象的属性(覆盖默认值)。"""
        for key, value in configs.items():
            setattr(self, key, value)


class AgentBase(ABC):
    """强化学习智能体的抽象基类, 规定了 get_action / update 接口并提供软更新、保存加载等通用工具。"""
    def __init__(
        self, config_type, configs: dict, verbose: bool = False,
        save_params: bool = False, load_params: bool = False
    ) -> None:
        """参数:
            config_type: 配置类(如 PPOConfig), 会用 configs 实例化。
            configs:     超参数 dict。
            verbose:     是否打印日志。
            save_params / load_params: 保存/加载时是否只处理参数(state_dict)而非整个对象。
        """
        self.device = device # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.configs = config_type(configs)
        self.check_list = []
        self.save_params = save_params
        self.load_params = load_params

    @abstractmethod
    def get_action(self, observation):
        """抽象方法: 根据观测返回动作。"""

    def _soft_update(self, target_net, current_net):
        """软更新目标网络: target = tau * current + (1 - tau) * target。
        目标网络缓慢跟随当前网络, 使价值估计的训练目标更稳定。
        """
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(current.data * self.configs.tau + target.data * (1. - self.configs.tau))
        
    def push_memory(self, observations):
        """把一条经验存入 self.memory。"""
        self.memory.push(observations)

    @abstractmethod
    def update(self):
        """抽象方法: 用缓存的经验更新网络参数。"""

    def epsilon_greedy(self, action, action_space, epsilon=0.1) -> Any:
        """以 epsilon 的概率返回随机动作, 否则返回原动作。"""
        if random.random() > epsilon:
            return action
        return action_space.sample()

    def lr_decay(self, lr: float, n_step: int, decay_type: str = "exp") -> float:
        """按线性或指数方式根据训练步数 n_step 衰减学习率。"""
        if decay_type == "linear":
            lr = lr * (1 - n_step / self.configs.max_train_steps)
        elif decay_type == "exp":
            lr = lr * np.exp(-n_step / self.configs.max_train_steps)
        return lr

    def explore(self, action, action_space):
        """按 explore_configs 中指定的探索方式处理动作(目前只支持 epsilon_greedy)。"""
        explore_configs = self.configs.explore_configs
        if explore_configs["type"] == "epsilon_greedy":
            return self.epsilon_greedy(action, action_space, explore_configs["epsilon"])
        return action

    def save(self, path: str = None, params_only: bool = None) -> None:
        """把模型保存到 path: params_only 时保存 check_list 中各项的 state_dict, 否则保存整个对象。"""
        if params_only is not None:
            self.save_params = params_only
        if self.save_params and len(self.check_list) > 0:
            checkpoint = dict()
            for name, item, save_state_dict in self.check_list:
                checkpoint[name] = item.state_dict() if save_state_dict else item
            torch.save(checkpoint, path)
        else:
            torch.save(self, path)
        
        if self.verbose:
            print("Save current model to %s" % path)

    def load(self, path: str = None, params_only: bool = None) -> None:
        """从 path 加载模型(子类有各自的重写版本, 实际使用的是子类实现)。"""
        if params_only is not None:
            self.load_params = params_only
        if self.load_params and len(self.check_list) > 0:
            checkpoint = torch.load(path)
            for name, item, save_state_dict in self.check_list:
                if save_state_dict:
                    item.load_state_dict(checkpoint[name])
                else:
                    item = checkpoint[name]
        else:
            torch.load(self, path)
        
            path =f"{path}/{name}_{id}.pth"
            state_dict = torch.load(path, map_location=self.device)
            object.load_state_dict(state_dict)

        if self.verbose:
            print("Load the model from %s" % path)
