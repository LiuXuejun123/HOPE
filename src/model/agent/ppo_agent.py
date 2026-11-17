"""PPO(Proximal Policy Optimization, 近端策略优化)智能体。

PPO 是 on-policy 的 actor-critic 算法:
    1. 用当前策略与环境交互, 收集 batch_size(8192) 条经验;
    2. 用 critic 和 GAE 估计每一步的优势(advantage), 即"这个动作比平均水平好多少";
    3. 在这批数据上训练 mini_epoch 轮, 用截断(clip)的概率比限制每次策略更新的幅度;
    4. 丢弃这批数据, 回到第 1 步。
本实现的策略是对角高斯分布: 均值由 actor 网络给出, log_std 是一个独立的可学习参数(与状态无关)。
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np

from model.agent_base import ConfigBase, AgentBase
from model.network import *
from model.replay_memory import ReplayMemory
from model.state_norm import StateNorm
from model.action_mask import ActionMask


class PPOConfig(ConfigBase):
    """PPO 的超参数。主要项:
        lr_actor / lr_critic: 学习率(critic 是 actor 的 5 倍);
        mini_epoch=10, mini_batch=32: 每批数据训练 10 轮, 每次用 32 条;
        clip_epsilon=0.2: 概率比截断范围 [0.8, 1.2];
        lambda_=0.95: GAE 的 λ;
        adv_norm / state_norm / use_gae 等: 常用训练技巧的开关。
    """
    def __init__(self, configs):
        """设置 PPO 默认超参数, 再用传入的 configs 覆盖。"""
        super().__init__()

        # hyperparameters
        self.lr_actor = self.lr
        self.lr_critic = self.lr*5
        self.adam_epsilon = 1e-8
        self.dist_type = "gaussian"
        self.hidden_size = 256
        self.mini_epoch = 10
        self.mini_batch = 32

        self.clip_epsilon = 0.2
        self.lambda_ = 0.95
        self.var_max = 1

        # tricks
        self.adv_norm = True
        self.state_norm = True
        self.reward_norm = False
        self.use_gae = True
        self.reward_scaling = False
        self.gradient_clip = False
        self.policy_entropy = False
        self.entropy_coef = 0.01

        self.merge_configs(configs)


class PPOAgent(AgentBase):
    """PPO 智能体, 包含 actor 网络、critic 网络、目标 critic、观测归一化和经验缓冲区。"""
    def __init__(
        self, configs: dict, discrete: bool = False, verbose: bool = False,
        save_params: bool = False, load_params: bool = False
    ) -> None:

        """参数:
            configs:  超参数 dict, 需包含 observation_shape、action_dim、actor_layers、critic_layers 等。
            discrete: 是否为离散动作空间(本项目为 False, 使用连续高斯策略)。
            verbose / save_params / load_params: 见 AgentBase。
        """
        super().__init__(PPOConfig, configs, verbose, save_params, load_params)
        self.discrete = discrete
        self.action_filter = ActionMask()

        # debug
        self.actor_loss_list = []
        self.critic_loss_list = []

        # the networks
        self._init_network()

        # As a on-policy RL algorithm, PPO does not have memory, the self.memory represents
        # the buffer
        self.memory = ReplayMemory(self.configs.batch_size, ["log_prob","next_obs"])

        # tricks
        if self.configs.state_norm:
            self.state_normalize = StateNorm(self.configs.observation_shape)

        
    def _init_network(self):
        """创建 actor 网络 + log_std 参数及其优化器、critic 网络及其优化器、目标 critic, 并登记需要保存的对象(check_list)。"""
        self.actor_net = \
            MultiObsEmbedding(self.configs.actor_layers).to(self.device)
        if self.configs.dist_type == "gaussian":
            self.log_std = \
                nn.Parameter(
                    torch.zeros(1, self.configs.action_dim), requires_grad=False
                ).to(self.device)
            self.log_std.requires_grad = True
            self.actor_optimizer = \
                torch.optim.Adam(
                    [{'params':self.actor_net.parameters()}, {'params': self.log_std}], 
                    self.configs.lr_actor, 
                    # eps=self.configs.lr_actor
                )
        else:
            self.actor_optimizer = \
                torch.optim.Adam(
                    self.actor_net.parameters(), 
                    eps=self.configs.lr_actor
                )

        self.critic_net = \
            MultiObsEmbedding(self.configs.critic_layers).to(self.device)
        self.critic_optimizer = \
            torch.optim.Adam(
                self.critic_net.parameters(), 
                self.configs.lr_critic,
                eps=self.configs.adam_epsilon
            )
        self.critic_target = deepcopy(self.critic_net).to(self.device)
        
        # save and load
        self.check_list = [ # (name, item, save_state_dict)
            ("configs", self.configs, 0),
            ("actor_net", self.actor_net, 1),
            ("actor_optimizer", self.actor_optimizer, 1),
            ("critic_net", self.critic_net, 1),
            ("critic_optimizer", self.critic_optimizer, 1),
            ("critic_target", self.critic_target, 1)
        ]
        if self.configs.dist_type == "gaussian":
            self.check_list.append(("log_std", self.log_std, 0))

    def _actor_forward(self, obs) -> torch.distributions.Distribution: # to be replaced
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
            if self.discrete:
                dist = Categorical(F.softmax(policy_dist, dim=1))
            elif self.configs.dist_type == "beta":
                alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                alpha = F.softplus(alpha) + 1.0
                beta = F.softplus(beta) + 1.0
                dist = Beta(alpha, beta)
            elif self.configs.dist_type == "gaussian":
                mean =  torch.clamp(policy_dist,-1,1)  
                log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
                std = torch.exp(log_std)
                dist = Normal(mean, std)
            else:
                raise NotImplementedError
            
        return dist
    
    def _post_process_action(self, action_dist:torch.distributions.Distribution , action_mask=None): # to be replaced
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
        """不使用动作掩码, 直接从高斯策略采样动作。

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

    def update(self): # to be replaced
        # convert batches to tensors

        # GAE computation cannot use shuffled data
        # batches = self.memory.shuffle()
        """用缓冲区中的一整批经验更新 actor 和 critic, 然后清空缓冲区。

        步骤:
            1. 按时间顺序取出全部数据(GAE 需要保持顺序, 不能打乱);
            2. TD 误差 delta = r + γ(1-done)V(s') - V(s), 再用 GAE 反向累加得到优势 adv;
               价值目标 v_target = adv + V_target(s), 并对 adv 做标准化;
            3. 重复 mini_epoch 次: 打乱数据, 按 mini_batch 切分, 对每个 mini-batch:
                   ratio = exp(新 log_prob - 旧 log_prob)
                   actor_loss  = -min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)
                   critic_loss = MSE(V(s), v_target)
               分别反向传播并更新; 每轮结束后软更新目标 critic。
        返回:
            (最后一个 mini-batch 的 actor_loss, critic_loss), 用于日志。
        """
        batches = self.memory.get_items(np.arange(len(self.memory)))
        state_batch = self.obs2tensor(batches["state"])
        
        if self.discrete:
            action_batch = torch.IntTensor(batches["action"]).to(self.device)
        else:
            action_batch = torch.FloatTensor(batches["action"]).to(self.device)
        rewards = torch.FloatTensor(np.array(batches["reward"])).unsqueeze(1)
        reward_batch = self._reward_norm(rewards) \
            if self.configs.reward_norm else rewards
        reward_batch = reward_batch.to(self.device)
        done_batch = torch.FloatTensor(batches["done"]).to(self.device).unsqueeze(1)
        old_log_prob_batch = torch.FloatTensor(batches["log_prob"]).to(self.device)
        next_state_batch = self.obs2tensor(batches["next_obs"])
        self.memory.clear()

        # GAE
        gae = 0
        adv = []

        with torch.no_grad():
            value = self.critic_net(state_batch)
            next_value = self.critic_net(next_state_batch)
            deltas = reward_batch + self.configs.gamma * (1 - done_batch) * next_value - value
            if self.configs.use_gae:
                for delta, done in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done_batch.cpu().flatten().numpy())):
                    gae = delta + self.configs.gamma * self.configs.lambda_ * gae * (1.0 - done)
                    adv.append(gae)
                adv.reverse()
                adv = torch.FloatTensor(adv).view(-1, 1).to(self.device)
            else:
                adv = deltas
            v_target = adv + self.critic_target(state_batch)
            if self.configs.adv_norm: # advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        
        # apply multi update epoch
        for _ in range(self.configs.mini_epoch):
            # use mini batch and shuffle data
            mini_batch = self.configs.mini_batch
            batchsize = self.configs.batch_size
            train_times = batchsize//mini_batch if batchsize%mini_batch==0 else batchsize//mini_batch+1
            random_idx = np.arange(batchsize)
            np.random.shuffle(random_idx)
            for i in range(train_times):
                if i == batchsize//mini_batch:
                    ri = random_idx[i*mini_batch:]
                else:
                    ri = random_idx[i*mini_batch:(i+1)*mini_batch]
                # state = state_batch[ri]
                state = self.get_obs(state_batch, ri)
                if self.discrete:
                    dist = Categorical(F.softmax(self.actor_net(state),dim=-1))
                    dist_entropy = dist.entropy().view(-1, 1)
                    log_prob= dist.log_prob(action_batch[ri].squeeze()).view(-1, 1)
                    old_log_prob = old_log_prob_batch[ri].view(-1,1)
                elif self.configs.dist_type == "beta":
                    policy_dist = self.actor_net(state)
                    alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                    alpha = F.softplus(alpha) + 1.0
                    beta = F.softplus(beta) + 1.0
                    dist = Beta(alpha, beta)
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    log_prob = dist.log_prob(action_batch[ri])
                    log_prob =torch.sum(log_prob,dim=1, keepdim=True)
                    old_log_prob =torch.sum(old_log_prob_batch[ri],dim=1, keepdim=True)
                elif self.configs.dist_type == "gaussian":
                    policy_dist = self.actor_net(state)
                    mean = torch.clamp(policy_dist, -1, 1)
                    log_std = self.log_std.expand_as(mean)
                    std = torch.exp(log_std)
                    dist = Normal(mean, std)
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    log_prob = dist.log_prob(action_batch[ri])
                    log_prob =torch.sum(log_prob,dim=1, keepdim=True)
                    old_log_prob =torch.sum(old_log_prob_batch[ri],dim=1, keepdim=True)
                prob_ratio = (log_prob - old_log_prob).exp()

                loss1 = prob_ratio * adv[ri]
                loss2 = torch.clamp(prob_ratio, 1 - self.configs.clip_epsilon, 1 + self.configs.clip_epsilon) * adv[ri]

                actor_loss = - torch.min(loss1, loss2)
                if self.configs.policy_entropy:
                    actor_loss += - self.configs.entropy_coef * dist_entropy
                critic_loss = F.mse_loss(v_target[ri], self.critic_net(state))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.mean().backward()
                critic_loss.mean().backward()
                
                self.actor_loss_list.append(actor_loss.mean().item())
                self.critic_loss_list.append(critic_loss.mean().item())
                if self.configs.gradient_clip: # gradient clip
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                    nn.utils.clip_grad_norm(self.actor_net.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step() 

            self._soft_update(self.critic_target, self.critic_net)

        if self.configs.lr_decay: # learning rate decay
            self.actor_optimizer.param_groups["lr"] = self.lr_decay(self.configs.lr_actor)
            self.critic_optimizer.param_groups["lr"] = self.lr_decay(self.configs.lr_critic)

        # for debug
        a = actor_loss.detach().cpu().numpy()[0][0]
        b = critic_loss.item()
        return a, b

    def save(self, path: str = None, params_only: bool = None) -> None: # to be replaced
        """保存模型。params_only=True 时保存一个 dict: check_list 中各网络/优化器的 state_dict,
        以及 log_std、观测归一化对象 state_norm、优化器对象; 否则保存整个 agent 对象。
        """
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
            checkpoint['optimizer'] = (self.actor_optimizer, self.critic_optimizer)
            torch.save(checkpoint, path)
        else:
            torch.save(self, path)
        
        if self.verbose:
            print("Save current model to %s" % path)

    def load(self, path: str = None, params_only: bool = None) -> None: # to be replaced
        """加载 save() 保存的模型(params_only=True 的格式), 包括网络参数、log_std、观测归一化统计量和优化器。
        注意: 观测归一化统计量必须和网络一起加载, 否则网络看到的输入分布会不一致。
        """
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
                self.actor_optimizer, self.critic_optimizer = checkpoint['optimizer']
        else:
            torch.load(self, path)
        
            path =f"{path}/{name}_{id}.pth"
            state_dict = torch.load(path, map_location=self.device)
            object.load_state_dict(state_dict)

        if self.verbose:
            print("Load the model from %s" % path)

    def load_actor(self, path: str = None) -> None: # to be replaced
        """只加载 actor 网络、log_std 和观测归一化(不加载 critic), 例如用于只做推理的场景。"""
        if len(self.check_list) > 0:
            checkpoint = torch.load(path, map_location=self.device)
            for name, item, save_state_dict in self.check_list:
                if name != 'actor_net':
                    continue
                if save_state_dict:
                    item.load_state_dict(checkpoint[name])
                else:
                    item = checkpoint[name]

            self.log_std.data.copy_(checkpoint['log']) 
            # self.actor_target_net = deepcopy(self.actor_net).to(self.device)
            self.state_normalize = checkpoint['state_norm']

    def load_img_encoder(self, path: str = None, require_grad: bool = False) -> None:
        """为 actor 和 critic 加载预训练的图像编码器, 并重新复制目标 critic。"""
        self.actor_net.load_img_encoder(path, self.device, require_grad)
        self.critic_net.load_img_encoder(path, self.device, require_grad)
        self.critic_target = deepcopy(self.critic_net).to(self.device)
        print('Load pretrained image encoder from path: %s'%path)
