"""车辆模型模块。

定义了三样东西:
    - Status:  一个 episode 在每一步结束后的状态(继续 / 到达 / 碰撞 / 出界 / 超时)。
    - State:   车辆某一时刻的位姿和速度(以后轴中心为参考点)。
    - KSModel / Vehicle: 运动学单车(自行车)模型, 根据动作 [转角, 速度] 推进车辆状态。
"""
from typing import Callable, List
from enum import Enum
import copy

import numpy as np
from shapely.geometry import Point, LinearRing
from shapely.affinity import affine_transform

from configs import *



class Status(Enum):
    """一个 episode 在每个 step 之后所处的状态。

    CONTINUE=继续行驶, ARRIVED=成功泊入, COLLIDED=发生碰撞,
    OUTBOUND=驶出地图边界, OUTTIME=超过最大步数(TOLERANT_TIME)。
    除 CONTINUE 外, 其他状态都会让 episode 结束(done=True)。
    """
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class State:
    """车辆在某一时刻的状态。

    属性:
        loc (Point):      后轴中心的位置 (x, y), 单位米。
        heading (float):  车头朝向, 弧度, 0 表示朝 x 轴正方向。
        speed (float):    速度(m/s), 负数表示倒车。
        steering (float): 前轮转角(rad)。
    """
    def __init__(self, raw_state: list):
        """用列表构造状态。

        参数:
            raw_state: [x, y, heading] 或 [x, y, heading, speed, steering]。
                       只给 3 个值时, 速度和转角默认为 0。
        """
        self.loc: Point = Point(raw_state[:2])
        self.heading: float = raw_state[2]
        if len(raw_state) == 3:
            self.speed: float = 0
            self.steering: float = 0
        else:
            self.speed: float = raw_state[3]
            self.steering: float = raw_state[4]

    def create_box(self) -> LinearRing:
        """计算车身矩形在世界坐标系下的轮廓。

        把以后轴中心为原点的车身模板 VehicleBox(见 configs.py) 按当前 heading 旋转,
        再平移到 loc, 返回 shapely 的 LinearRing(4 个角点, 顺序: 右后、右前、左前、左后)。
        碰撞检测、渲染、到达判定都依赖这个矩形。
        """
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        return affine_transform(VehicleBox, mat)

    def get_pos(self,):
        """返回 (x, y, heading) 三元组, 常用于传给 Reeds-Shepp 曲线计算。"""
        return (self.loc.x, self.loc.y, self.heading)


class KSModel(object):
    """运动学单车模型(Kinematic Single-Track / Bicycle Model)。

    输入当前状态和动作 [前轮转角, 速度], 用欧拉积分推算一小段时间后的位置和朝向:
        x'   = v * cos(heading)
        y'   = v * sin(heading)
        heading' = v * tan(steer) / wheel_base
    注意这里的动作直接设定速度和转角(不是加速度), 即速度可以瞬间改变。
    以后轴中心为局部坐标原点。
    """
    def __init__(
        self, 
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list
    ):
        """参数:
            wheel_base:  轴距(米)。
            step_len:    每个物理仿真步的时长(秒)。
            n_step:      一个环境 step 默认包含的物理仿真步数。
            speed_range: 速度上下限 [min, max]。
            angle_range: 前轮转角上下限 [min, max]。
        """
        self.wheel_base = wheel_base
        self.step_len = step_len
        self.n_step = n_step
        self.speed_range = speed_range
        self.angle_range = angle_range
        self.mini_iter = 20


    def step(self, state: State, action: list, step_time:int=NUM_STEP) -> State:
        """按动作推进车辆状态, 返回一个新的 State(不修改输入)。

        参数:
            state:     当前状态。
            action:    [steer, speed], 已经是真实物理量(rad, m/s), 会被裁剪到合法范围。
            step_time: 推进多少个物理仿真步; 每步内部再细分为 mini_iter=20 次积分以提高精度。

        返回:
            新的 State。总行驶时间 = step_time * step_len 秒。
        """
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, speed = action
        new_state.steering = steer
        new_state.speed = speed
        new_state.speed = np.clip(new_state.speed, *self.speed_range)
        new_state.steering = np.clip(new_state.steering, *self.angle_range)

        for _ in range(step_time):
            for _ in range(self.mini_iter):
                x += new_state.speed * np.cos(new_state.heading) * self.step_len/self.mini_iter
                y += new_state.speed * np.sin(new_state.heading) * self.step_len/self.mini_iter
                new_state.heading += \
                    new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len/self.mini_iter 

        new_state.loc = Point(x, y)
        return new_state


class Vehicle:
    """被控制的自车。

    封装了当前状态 state、车身矩形 box、历史轨迹 trajectory 和运动学模型,
    对外提供 reset / step / retreat(回退一步) 三个操作。
    """
    def __init__(
        self,
        wheel_base: float = WHEEL_BASE,
        step_len: float = STEP_LENGTH,
        n_step: int = NUM_STEP,
        speed_range: list = VALID_SPEED, 
        angle_range: list = VALID_STEER
    ) -> None:

        """参数含义同 KSModel; 默认值都取自 configs.py。"""
        self.initial_state: list = None
        self.state: State = None
        self.box: LinearRing = None
        self.trajectory: List[State] = []
        self.kinetic_model: Callable = \
            KSModel(wheel_base, step_len, n_step, speed_range, angle_range)
        self.color = COLOR_POOL[0]
        self.v_max = None
        self.v_min = None

    def reset(self, initial_state: State):
        """把车放到初始状态并清空轨迹。

        参数:
            initial_state (State): 起点状态, 一般由地图的 reset() 给出。
        """
        self.initial_state = initial_state
        self.state = self.initial_state
        # self.color = random.sample(COLOR_POOL, 1)[0]
        self.v_max = self.initial_state.speed
        self.v_min = self.initial_state.speed
        self.box = self.state.create_box()
        self.trajectory.clear()
        self.trajectory.append(self.state)
        self.tmp_trajectory = self.trajectory.copy()

    def step(self, action: np.ndarray, step_time: int=NUM_STEP):
        """执行一个动作, 更新状态、车身矩形和轨迹。

        参数:
            action:    [steer, speed] 真实物理量。
            step_time: 物理仿真步数(环境里每次传 1, 以便逐小步检测碰撞)。

        返回:
            prev_info: 执行前的 (state, box, v_max, v_min), 可交给 retreat() 撤销这一步。
        """
        prev_info = copy.deepcopy((self.state, self.box, self.v_max, self.v_min))
        self.state = self.kinetic_model.step(self.state, action, step_time)
        self.box = self.state.create_box()
        self.trajectory.append(self.state)
        self.tmp_trajectory.append(self.state)
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min
        return prev_info

    def retreat(self, prev_info):
        """撤销最近一次 step(): 恢复到 prev_info 记录的状态, 并删除轨迹最后一个点。
        环境在检测到"这一小步会撞上障碍物"时调用, 让车停在碰撞前的位置。
        """
        self.state, self.box, self.v_max, self.v_min = prev_info
        self.trajectory.pop(-1)
