"""SAC(Soft Actor-Critic)智能体。

SAC 是 off-policy 的最大熵 actor-critic 算法:
    - 经验存入回放池, 每次随机采样一小批训练, 数据可重复使用;
    - 两个 Q 网络取较小值来缓解价值高估, 各自配一个软更新的目标网络;
    - 目标函数中加入策略熵(鼓励探索), 熵的权重 alpha 自动调节。
注意这里的策略 std 与 PPO 一样是独立参数 log_std, 动作做了 clamp 而非 tanh 压缩。
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from model.agent_base import ConfigBase, AgentBase
from model.network import *
from model.replay_memory import ReplayMemory
from model.state_norm import StateNorm
from model.action_mask import ActionMask

class SACCriticAdapter(nn.Module):
    """SAC 的 Q 网络: Q(s, a)。在 MultiObsEmbedding 基础上把动作 a 作为额外一个模态(token)输入。"""
    def __init__(self, configs: dict, action_dim:int=2):
        """在 critic 配置中加入 input_action_dim 并把模态数加 1, 再构建 MultiObsEmbedding。"""
        super().__init__()
        self.configs = deepcopy(configs)
        self.configs['input_action_dim'] = action_dim
        self.configs['n_modal'] += 1
        self.net = MultiObsEmbedding(self.configs)

    def forward(self, state: dict, action: torch.Tensor) -> torch.Tensor:
        """参数:
            state:  观测张量 dict(会被原地加入 'action' 键)。
            action: (B, action_dim) 动作张量。
        返回:
            (B, 1) 的 Q 值。
        """
        state_action = state
        state_action['action'] = action
        x = self.net(state_action)
        return x
    
    def load_img_encoder(self, path: str = None, device: str = None, require_grad: bool = False) -> None:
        """加载预训练图像编码器。"""
        self.net.load_img_encoder(path, device, require_grad)


class SACConfig(ConfigBase):
    """SAC 的超参数。主要项:
        memory_size=10240: 回放池大小(也是开始训练前随机探索的步数);
        batch_size=32: 每次更新采样的数量;
        tau=0.005: 目标网络软更新系数;
        initial_temperature=0.01: 熵权重 alpha 的初值; target_entropy=-action_dim: 目标熵;
        learn_temperature: 是否自动调节 alpha。
    """
    def __init__(self, configs):
        """设置 SAC 默认超参数, 再用传入的 configs 覆盖。"""
        super().__init__()

        # hyperparameters
        self.lr_actor = self.lr
        self.lr_critic = self.lr
        self.lr_alpha = self.lr
        self.tau = 0.005
        self.adam_epsilon = 1e-8
        self.dist_type = "gaussian"
        self.hidden_size = 256
        self.memory_size = 10240
        self.batch_size = 32
        # self.mini_batch_size = 32
        self.mini_epoch = 1
        self.initial_temperature = 0.01
        self.action_dim = 2
        self.target_entropy = -self.action_dim

        # tricks
        self.learn_temperature = True
        self.state_norm = True
        self.reward_norm = False
        self.reward_scaling = False

        self.merge_configs(configs)


class SACAgent(AgentBase):
    """SAC 智能体, 包含 actor + log_std、两个 Q 网络及其目标网络、可学习的 log_alpha、回放池和观测归一化。"""
    def __init__(
        self, configs: dict, discrete: bool = False, verbose: bool = False,
        save_params: bool = False, load_params: bool = False
    ) -> None:

        """参数:
            configs:  超参数 dict, 需包含 observation_shape、action_dim、actor_layers、critic_layers 等。
            discrete: 是否为离散动作空间(本项目为 False, 使用连续高斯策略)。
            verbose / save_params / load_params: 见 AgentBase。
        """
        super().__init__(SACConfig, configs, verbose, save_params, load_params)
        self.discrete = discrete
        self.action_filter = ActionMask()

        # debug
        self.actor_loss_list = []
        self.critic_loss_list = []

        # the networks
        self._init_network()

        # As a on-policy RL algorithm, PPO does not have memory, the self.memory represents
        # the buffer
        self.memory = ReplayMemory(self.configs.memory_size, ["log_prob","next_obs"])

        # tricks
        if self.configs.state_norm:
            self.state_normalize = StateNorm(self.configs.observation_shape)

        
    def _init_network(self):
        """创建 actor、两个 Q 网络及目标网络、log_alpha 及各自的优化器, 并登记 check_list。"""

        ## actor net
        self.actor_net = MultiObsEmbedding(self.configs.actor_layers).to(self.device)
        self.log_std = \
            nn.Parameter(
                -torch.zeros(1, self.configs.action_dim), requires_grad=False
            ).to(self.device)
        self.log_std.requires_grad = True
        self.actor_optimizer = \
            torch.optim.Adam(
                [{'params':self.actor_net.parameters()}, {'params': self.log_std}], 
                self.configs.lr_actor, 
            )

        ## critic net
        self.critic_net1 = SACCriticAdapter(self.configs.critic_layers).to(self.device)
        self.critic_target_net1 = deepcopy(self.critic_net1)
        self.critic_optimizer1 = torch.optim.Adam(self.critic_net1.parameters(), self.configs.lr_critic)

        self.critic_net2 = SACCriticAdapter(self.configs.critic_layers).to(self.device)
        self.critic_target_net2 = deepcopy(self.critic_net2)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), self.configs.lr_critic)

        ## alpha
        self.log_alpha = torch.tensor(np.log(self.configs.initial_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], self.configs.lr_alpha)

        
        # save and load
        self.check_list = [ # (name, item, save_state_dict)
            ("configs", self.configs, 0),
            ("actor_net", self.actor_net, 1),
            ("actor_optimizer", self.actor_optimizer, 1),
            ("critic_net1", self.critic_net1, 1),
            ("critic_optimizer1", self.critic_optimizer1, 1),
            ("critic_target1", self.critic_target_net1, 1),
            ("critic_net2", self.critic_net2, 1),
            ("critic_optimizer2", self.critic_optimizer2, 1),
            ("critic_target2", self.critic_target_net2, 1),
            ("log_alpha", self.log_alpha, 0),
            ("log_alpha_optimizer", self.log_alpha_optimizer, 1),
            ("log_std", self.log_std, 0)
        ]

    def _actor_forward(self, obs) -> torch.distributions.Distribution:
        """用 actor 网络计算单个观测下的动作分布(不计算梯度)。

        参数:
            obs: 原始观测 dict(内部会先归一化、再转张量)。
        返回:
            torch 分布对象; 默认为 Normal(mean=clamp(actor(obs), -1, 1), std=exp(log_std))。
        """
        observation = deepcopy(obs)
        if self.configs.state_norm:
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        
        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            if len(policy_dist.shape) > 1 and policy_dist.shape[0] > 1:
                raise NotImplementedError
            mean =  torch.clamp(policy_dist,-1,1)  
            log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            
        return dist
    
    def _post_process_action(self, action_dist:torch.distributions.Distribution , action_mask=None):
        """从分布中得到最终动作及其对数概率。

        参数:
            action_dist: _actor_forward 给出的分布。
            action_mask: 若给出, 用 ActionMask.choose_action 在 42 个离散安全动作中按概率采样;
                         否则直接从高斯分布采样。
        返回:
            action (np.ndarray, 归一化到 [-1, 1]), log_prob (np.ndarray, 该动作在高斯分布下的对数概率, 每维一个值)。
        """
        if action_mask is not None:
            mean, std = action_dist.mean, action_dist.stddev
            action = self.action_filter.choose_action(mean, std, action_mask)
            action = torch.FloatTensor(action).to(self.device)
        else:
            action = action_dist.sample()

        if not self.discrete and self.configs.dist_type == "gaussian":
                action = torch.clamp(action, -1, 1)
        log_prob = action_dist.log_prob(action)
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return action, log_prob

    def choose_action(self, obs):

        """训练和评估时使用的动作选择: 使用动作掩码, 只在安全动作中采样。

        参数:
            obs: 观测 dict(需包含 'action_mask')。
        返回:
            (action, log_prob)。
        """
        dist = self._actor_forward(obs)
        action_mask = obs['action_mask']
        action, other_info = self._post_process_action(dist, action_mask)

        return action, other_info

    def get_action(self, obs: np.ndarray):
        """不使用动作掩码, 直接从高斯策略采样动作(SAC 训练和评估默认都走这个接口)。

        参数:
            obs: 观测 dict。
        返回:
            action (np.ndarray, [-1,1]), log_prob (np.ndarray)。
        """
        dist = self._actor_forward(obs)
        action, log_prob = self._post_process_action(dist)
                
        return action, log_prob

    def get_log_prob(self, obs: np.ndarray, action: np.ndarray):
        """计算给定动作在当前策略下的对数概率。
        RS 规划器接管时用它为规划器给出的动作计算 log_prob, 使这些数据也能放进 PPO 缓冲区。

        参数:
            obs:    观测 dict。
            action: 归一化动作。
        返回:
            log_prob (np.ndarray)。
        """
        dist = self._actor_forward(obs)
        
        action = torch.FloatTensor(action).to(self.device)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return log_prob

    def push_memory(self, observations):
        """存入一条经验 (obs, action, reward, done, log_prob, next_obs)。
        存入前对 obs、next_obs 做观测归一化, 并用 next_obs 更新归一化统计量。
        """
        obs, action, reward, done, log_prob, next_obs = deepcopy(observations)
        if self.configs.state_norm:
            obs = self.state_normalize.state_norm(obs)
            next_obs = self.state_normalize.state_norm(next_obs,update=True)
        observations = (obs, action, reward, done, log_prob, next_obs)
        self.memory.push(observations)

    def _reward_norm(self, reward):
        """把一批奖励标准化为均值 0、标准差 1(默认关闭)。"""
        return (reward - reward.mean()) / (reward.std() + 1e-8)

    def obs2tensor(self, obs):
        """把观测转换为张量 dict。

        参数:
            obs: 单个观测 dict(会加一个 batch 维) 或 观测 dict 的列表(会按模态堆叠成一个 batch)。
        返回:
            {模态名: FloatTensor}, 已放到 self.device 上。
        """
        if isinstance(obs, list):
            merged_obs = {}
            for obs_type in self.configs.observation_shape.keys():
                merged_obs[obs_type] = []
                for o in obs:
                    merged_obs[obs_type].append(o[obs_type])
                merged_obs[obs_type] = torch.FloatTensor(np.array(merged_obs[obs_type])).to(self.device)
            obs = merged_obs 
        elif isinstance(obs, dict):
            for obs_type in self.configs.observation_shape.keys():
                obs[obs_type] = torch.FloatTensor(obs[obs_type]).to(self.device).unsqueeze(0)
        else:
            raise NotImplementedError()
        return obs
    
    def get_obs(self, obs, ids):
        """从 batch 观测 dict 中按下标 ids 取出一个 mini-batch。"""
        return {k:obs[k][ids] for k in obs }
    
    def _merge_state_action(self, state:dict, action:torch.tensor):
        """把动作加入观测 dict 的 'action' 键(当前未使用)。"""
        state['action'] = action
        return state
    
    @property
    def alpha(self):
        """熵权重 alpha = exp(log_alpha)。"""
        return self.log_alpha.exp()
    
    def _get_action_and_log_prob(self, obs):
        """对一批观测张量用重参数化(rsample)采样动作, 返回 (action, log_prob)。
        重参数化使梯度可以从 Q 值经过动作反传到 actor。
        """
        action_policy = self.actor_net(obs)
        mean =  torch.clamp(action_policy,-1,1)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        action_dist = Normal(mean, std)
        action_batch = action_dist.rsample()
        
        action_batch = torch.clamp(action_batch, -1, 1)
        log_prob = action_dist.log_prob(action_batch)
        return action_batch, log_prob

    def update(self):
        """从回放池采样一批数据, 更新一次网络。

        步骤:
            1. Q 目标: y = r + γ(1-done) * [min(Q1', Q2')(s', a') - alpha * log π(a'|s')], a' 由当前策略采样;
            2. 两个 Q 网络分别最小化 MSE(Q(s,a), y);
            3. 冻结 Q 网络, actor 最小化 alpha * log π(a|s) - min(Q1, Q2)(s, a);
            4. 若开启, 调节 alpha 使策略熵接近 target_entropy;
            5. 软更新两个目标 Q 网络。
        返回:
            (actor_loss, q1_loss), 用于日志。
        """
        for _ in range(self.configs.mini_epoch):
            batches = self.memory.sample(self.configs.batch_size)
            state_batch = self.obs2tensor(batches["state"])
            action_batch = torch.FloatTensor(batches["action"]).to(self.device)
            rewards = torch.FloatTensor(np.array(batches["reward"])).unsqueeze(1)
            reward_batch = self._reward_norm(rewards) \
                if self.configs.reward_norm else rewards
            reward_batch = reward_batch.to(self.device)
            done_batch = torch.FloatTensor(batches["done"]).to(self.device).unsqueeze(1)
            next_state_batch = self.obs2tensor(batches["next_obs"])
            
            # soft Q loss
            with torch.no_grad():
                next_action_batch, next_log_prob = self._get_action_and_log_prob(next_state_batch)
                next_log_prob = next_log_prob.sum(-1, keepdim=True)
                q1_target = self.critic_target_net1(next_state_batch, next_action_batch)
                q2_target = self.critic_target_net2(next_state_batch, next_action_batch)
                q_target = reward_batch + (1 - done_batch) * self.configs.gamma * (
                    torch.min(q1_target, q2_target) - self.alpha.detach() * next_log_prob
                )

            current_q1 = self.critic_net1(state_batch, action_batch)
            current_q2 = self.critic_net2(state_batch, action_batch)
            q1_loss = F.mse_loss(current_q1, q_target.detach())
            q2_loss = F.mse_loss(current_q2, q_target.detach())

            # update the critic networks
            self.critic_optimizer1.zero_grad()
            q1_loss.backward()
            self.critic_optimizer1.step()
            self.critic_optimizer2.zero_grad()
            q2_loss.backward()
            self.critic_optimizer2.step()

            # freeze critic network
            for params in self.critic_net1.parameters():
                params.requires_grad = False
            for params in self.critic_net2.parameters():
                params.requires_grad = False

            # policy loss
            action_, log_prob = self._get_action_and_log_prob(state_batch)
            log_prob = log_prob.sum(-1, keepdim=True)
            q1_value = self.critic_net1(state_batch, action_)
            q2_value = self.critic_net2(state_batch, action_)
            actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_value, q2_value)).mean()

            # update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # unfreeze critic network
            for params in self.critic_net1.parameters():
                params.requires_grad = True
            for params in self.critic_net2.parameters():
                params.requires_grad = True

            # optimize alpha
            if self.configs.learn_temperature:
                alpha_loss = (self.alpha * (-log_prob - self.configs.target_entropy).detach()).mean()
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

            # soft update target networks
            self._soft_update(self.critic_target_net1, self.critic_net1)
            self._soft_update(self.critic_target_net2, self.critic_net2)


        # for debug
        a = actor_loss.detach().cpu().numpy()
        b = q1_loss.item()
        return a, b

    def save(self, path: str = None, params_only: bool = None) -> None:
        """保存模型(格式同 PPOAgent.save, 优化器元组包含 actor 和两个 critic 的优化器)。"""
        if params_only is not None:
            self.save_params = params_only
        if self.save_params and len(self.check_list) > 0:
            checkpoint = dict()
            for name, item, save_state_dict in self.check_list:
                checkpoint[name] = item.state_dict() if save_state_dict else item
            # for PPO extra save
            if self.configs.dist_type == "gaussian":
                checkpoint['log'] = self.log_std
            checkpoint['state_norm'] = self.state_normalize # (self.state_mean, self.state_std, self.S, self.n_state)
            checkpoint['optimizer'] = (self.actor_optimizer, self.critic_optimizer1, self.critic_optimizer2)
            torch.save(checkpoint, path)
        else:
            torch.save(self, path)
        
        if self.verbose:
            print("Save current model to %s" % path)

    def load(self, path: str = None, params_only: bool = None) -> None:
        """加载 save() 保存的模型, 包括网络参数、log_std、观测归一化统计量和优化器。"""
        if params_only is not None:
            self.load_params = params_only
        if self.load_params and len(self.check_list) > 0:
            checkpoint = torch.load(path, map_location=self.device)
            for name, item, save_state_dict in self.check_list:
                if save_state_dict:
                    item.load_state_dict(checkpoint[name])
                else:
                    item = checkpoint[name]

            self.log_std.data.copy_(checkpoint['log']) 
            
            self.state_normalize = checkpoint['state_norm'] 
            if 'optimizer' in checkpoint.keys():
                self.actor_optimizer, self.critic_optimizer1, self.critic_optimizer2 = checkpoint['optimizer']
        else:
            torch.load(self, path)
        
            path =f"{path}/{name}_{id}.pth"
            state_dict = torch.load(path, map_location=self.device)
            object.load_state_dict(state_dict)

        if self.verbose:
            print("Load the model from %s" % path)

    def load_img_encoder(self, path: str = None, require_grad: bool = False) -> None:
        """为 actor 和两个 Q 网络加载预训练图像编码器, 并重新复制目标网络。"""
        self.actor_net.load_img_encoder(path, self.device, require_grad)
        self.critic_net1.load_img_encoder(path, self.device, require_grad)
        self.critic_target_net1 = deepcopy(self.critic_net1).to(self.device)
        self.critic_net2.load_img_encoder(path, self.device, require_grad)
        self.critic_target_net2 = deepcopy(self.critic_net2).to(self.device)
        print('Load pretrained image encoder from path: %s'%path)
