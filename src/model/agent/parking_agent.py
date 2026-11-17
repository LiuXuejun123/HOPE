"""混合策略(Hybrid Policy), 即 HOPE 名字中的 "Hybrid"。

ParkingAgent 把 RL 智能体和 Reeds-Shepp 规划器组合起来:
    - 平时由 RL 智能体(PPO/SAC)决策;
    - 一旦环境报告找到了一条到终点的无碰撞 RS 曲线(info['path_to_dest']),
      就切换为 RsPlanner 逐步执行这条曲线, 直到执行完毕。
"""

class RsPlanner(object):
    """把一条 RS 曲线转换成环境动作序列并逐个输出的执行器。"""
    def __init__(self, step_ratio:float) -> None:
        """参数:
            step_ratio: 一个环境 step 以最大速度能行驶的距离(米),
                        = 物理步长 * 每步物理步数 * 最大速度 = 0.05 * 10 * 2.5 = 1.25。
        """
        self.route = None
        self.actions = []
        self.step_ratio = step_ratio

    def reset(self,):
        """清空当前路径和动作队列, 交还控制权给 RL 智能体。"""
        self.route = None
        self.actions.clear()
    
    def set_rs_path(self, rs_path):
        """把 RS 路径转换成归一化动作序列。

        每一段 L/R/S 映射为转角 1/-1/0(即最大左转/最大右转/直行, 归一化后),
        按 step_ratio 换算成需要多少个 step; 每个 step 的速度为 ±1(满速前进/倒车),
        最后不足一步的余量用较小的速度表示。
        """
        action_type = {'L':1, 'S':0, 'R':-1}
        self.route = rs_path
        step_ratio = self.step_ratio
        action_list = []
        for i in range(len(rs_path.ctypes)):
            steer = action_type[rs_path.ctypes[i]]
            step_len = rs_path.lengths[i]/step_ratio
            action_list.append([steer, step_len])

        # divide the action
        filtered_actions = []
        for action in action_list:
            action[0] *= 1
            if abs(action[1])<1 and abs(action[1])>1e-3:
                filtered_actions.append(action)
            elif action[1]>1:
                while action[1]>1:
                    filtered_actions.append([action[0], 1])
                    action[1] -= 1
                if abs(action[1])>1e-3:
                    filtered_actions.append(action)
            elif action[1]<-1:
                while action[1]<-1:
                    filtered_actions.append([action[0], -1])
                    action[1] += 1
                if abs(action[1])>1e-3:
                    filtered_actions.append(action)
        
        self.actions = filtered_actions

    def get_action(self, ):
        """弹出下一个动作; 动作队列用完时自动 reset。"""
        action = self.actions.pop(0)
        if len(self.actions) == 0 and self.route is not None:
            self.reset()
        return action

class ParkingAgent(object):
    """RL 智能体 + RS 规划器的组合。没有定义的属性/方法会通过 __getattr__ 转发给内部的 RL 智能体
    (所以训练脚本里可以直接写 parking_agent.memory、parking_agent.configs、parking_agent.log_std)。
    """
    def __init__(
        self, rl_agent, planner=None,
    ) -> None:
        """rl_agent: PPOAgent 或 SACAgent; planner: RsPlanner, 为 None 时退化为纯 RL。"""
        self.agent = rl_agent
        self.planner = planner

    def __getattr__(self, name: str):
        """把未定义的公开属性访问转发给内部 RL 智能体。"""
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)
    
    def reset(self,):
        """新 episode 开始时清空规划器状态。"""
        if self.planner is not None:
            self.planner.reset()

    def set_planner_path(self, path=None, forced=False):
        """设置 RS 路径。只有当规划器当前空闲(或 forced=True)时才接受新路径,
        即一旦开始执行某条 RS 曲线, 就会把它执行完, 不会被后续找到的路径打断。
        """
        if self.planner is None:
            return
        if path is not None and (forced or self.planner.route is None):
            self.planner.set_rs_path(path)

    @property
    def executing_rs(self,):
        """当前是否正在执行 RS 路径。"""
        return not (self.planner is None or self.planner.route is None)
    
    def get_log_prob(self, obs, action):
        """转发给 RL 智能体的 get_log_prob。"""
        return self.agent.get_log_prob(obs, action)

    def choose_action(self, obs):
        """获取融合后的决策(使用动作掩码版本)。

        参数:
            obs: 观测 dict。
        返回:
            action: 规划器在执行时为 RS 动作, 否则为 RL 智能体经动作掩码采样的动作;
            other:  该动作在 RL 策略下的 log_prob(PPO 训练需要)。
        """
        if not self.executing_rs:
            return self.agent.choose_action(obs)
        else:
            action = self.planner.get_action()
            log_prob = self.agent.get_log_prob(obs, action)
            return action, log_prob
        
    def get_action(self, obs):
        """获取融合后的决策(不使用动作掩码采样的版本), 返回值同 choose_action。"""
        if not self.executing_rs:
            return self.agent.get_action(obs)
        else:
            action = self.planner.get_action()
            log_prob = self.agent.get_log_prob(obs, action)
            return action, log_prob
            
    def push_memory(self, experience):
        """转发给 RL 智能体存储经验(包括 RS 规划器执行时产生的经验)。"""
        self.agent.push_memory(experience)

    def update(self,):
        """转发给 RL 智能体更新网络。"""
        return self.agent.update()
    
    def save(self, *args, **kwargs ):
        """转发给 RL 智能体保存模型。"""
        self.agent.save(*args, **kwargs )

    def load(self, *args, **kwargs ):
        """转发给 RL 智能体加载模型。"""
        self.agent.load(*args, **kwargs)
