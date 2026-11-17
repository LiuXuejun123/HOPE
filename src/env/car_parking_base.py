"""泊车强化学习环境(gym.Env), 整个项目的核心仿真环境。

每个 step: 执行动作 -> 物理仿真并逐小步检测碰撞 -> 渲染并生成观测 -> 判定状态 -> 计算奖励分量
-> (距离终点足够近时)尝试搜索一条无碰撞的 Reeds-Shepp 曲线直达终点。

注意: 这是一个"不会因碰撞结束"的环境(ENV_COLLIDE=False)。如果某一小步会撞上障碍物,
车辆会回退到碰撞前的位置并停下, 而不是返回 COLLIDED。
"""


import sys
sys.path.append("../")
from typing import Optional, Union
import math
from typing import OrderedDict
import random

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from heapdict import heapdict
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from env.vehicle import *
from env.map_base import *
from env.lidar_simulator import LidarSimlator
from env.parking_map_normal import ParkingMapNormal
from env.parking_map_dlp import ParkingMapDLP
import env.reeds_shepp as rsCurve
from env.observation_processor import Obs_Processor
from model.action_mask import ActionMask
from configs import *

class CarParking(gym.Env):

    """泊车环境。

    动作空间: Box([转角, 速度]), 转角 ∈ VALID_STEER, 速度 ∈ VALID_SPEED。
    观测空间(dict):
        'lidar':       (120,)     激光雷达, 车身到障碍物的距离;
        'target':      (5,)       目标位置表示 [距离, cos(方位角), sin(方位角), cos(相对朝向), cos(相对朝向)];
        'img':         (64,64,3)  以自车为中心、车头朝上的鸟瞰图;
        'action_mask': (42,)      每个离散动作可安全行驶的比例 [0,1]。
    step() 返回的不是标量奖励而是奖励分量, 由 env_wrapper.reward_shaping 加权合成。
    """
    metadata = {
        "render_mode": [
            "human", 
            "rgb_array",
        ]
    }

    def __init__(
        self, 
        render_mode: str = None,
        fps: int = FPS,
        verbose: bool =True, 
        use_lidar_observation: bool =USE_LIDAR,
        use_img_observation: bool=USE_IMG,
        use_action_mask: bool=USE_ACTION_MASK,
    ):
        """参数:
            render_mode: "human" 弹窗显示; "rgb_array" 隐藏窗口(训练时更快)。
            fps:         渲染帧率上限。
            verbose:     是否打印信息。
            use_lidar_observation / use_img_observation / use_action_mask:
                         是否在观测中包含对应模态, 默认值来自 configs.py。
        根据 MAP_LEVEL 创建地图(随机地图或 DLP 真实地图), 并定义动作空间和观测空间。
        """
        super().__init__()

        self.verbose = verbose
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        self.use_action_mask = use_action_mask
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.t = 0.0
        self.k = None
        self.level = MAP_LEVEL
        self.tgt_repr_size = 5 # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)

        if self.level in ['Normal', 'Complex', 'Extrem']:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0

        self.action_space = spaces.Box(
            np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
            np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
        ) # steer, speed
       
        self.observation_space = {}
        if self.use_action_mask:
            self.action_filter = ActionMask()
            self.observation_space['action_mask'] = spaces.Box(low=0, high=1, 
                shape=(N_DISCRETE_ACTION,), dtype=np.float64
            )
        if self.use_img_observation:
            self.img_processor = Obs_Processor()
            self.observation_space['img'] = spaces.Box(low=0, high=255, 
                shape=(OBS_W//self.img_processor.downsample_rate, OBS_H//self.img_processor.downsample_rate, 
                self.img_processor.n_channels), dtype=np.uint8
            )
            self.raw_img_shape = (OBS_W, OBS_H, 3)
        if self.use_lidar_observation:
            # the observation is composed of lidar points and target representation
            # the target representation is (relative_distance, cos(theta), sin(theta), cos(phi), sin(phi))
            # where the theta indicates the relative angle of parking lot, and phi means the heading of 
            # parking lot in the polar coordinate of the ego car's view
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM))*LIDAR_RANGE
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM,), dtype=np.float64
            )
        low_bound = np.array([0,-1,-1,-1,-1])
        high_bound = np.array([MAX_DIST_TO_DEST,1,1,1,1])
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float64
        )
    
    def set_level(self, level:str=None):
        """切换场景难度/类型。level 取 "Normal" / "Complex" / "Extrem" / "dlp";
        为 None 时使用默认的 ParkingMapNormal。会重新创建地图对象。
        """
        if level is None:
            self.map = ParkingMapNormal()
            return
        self.level = level
        if self.level in ['Normal', 'Complex', 'Extrem',]:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()

    def reset(self, case_id: int = None, data_dir: str = None, level: str = None,) -> np.ndarray:
        """开始一个新 episode。

        参数:
            case_id:  场景编号(含义见各地图类的 reset)。
            data_dir: DLP 场景文件路径(可选)。
            level:    若给出, 先切换难度。
        返回:
            初始观测 dict(通过 step(None) 获取, 不执行动作)。
        """
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0
        self.t = 0.0

        if level is not None:
            self.set_level(level)
        initial_state = self.map.reset(case_id, data_dir)
        self.vehicle.reset(initial_state)
        self.matrix = self.coord_transform_matrix()
        return self.step()[0]

    def coord_transform_matrix(self) -> list:
        """计算世界坐标 -> pygame 屏幕坐标的仿射矩阵 [k, 0, 0, k, bx, by]。
        k=K 为缩放比例(像素/米), bx, by 使地图居中显示。
        """
        k = K
        bx = 0.5 * (WIN_W - k * (self.map.xmax + self.map.xmin))
        by = 0.5 * (WIN_H - k * (self.map.ymax + self.map.ymin))
        self.k = k
        return [k, 0, 0, k, bx, by]

    def _coord_transform(self, object) -> list:
        """把 shapely 几何体变换到屏幕坐标, 返回坐标点列表(用于 pygame 绘制)。"""
        transformed = affine_transform(object, self.matrix)
        return list(transformed.coords)

    def _detect_collision(self):
        # return False
        """车身矩形与任一障碍物相交则返回 True。"""
        for obstacle in self.map.obstacles:
            if self.vehicle.box.intersects(obstacle.shape):
                return True
        return False
    
    def _detect_outbound(self):
        """后轴中心超出地图边界则返回 True。"""
        x, y = self.vehicle.state.loc.x, self.vehicle.state.loc.y
        return x>self.map.xmax or x<self.map.xmin or y>self.map.ymax or y<self.map.ymin

    def _check_arrived(self):
        """车身与目标车位的重叠面积超过车位面积的 95% 时视为泊车成功。"""
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.95:
            return True
        return False
    
    def _check_time_exceeded(self):
        """步数超过 TOLERANT_TIME(200) 时返回 True。"""
        return self.t > TOLERANT_TIME

    def _check_status(self):
        """按 碰撞 > 出界 > 到达 > 超时 的优先级判定当前状态, 都不满足则 CONTINUE。"""
        if self._detect_collision():
            return Status.COLLIDED
        if self._detect_outbound():
            return Status.OUTBOUND
        if self._check_arrived():
            return Status.ARRIVED
        if self._check_time_exceeded():
            return Status.OUTTIME
        return Status.CONTINUE

    def _get_reward(self, prev_state: State, curr_state: State):

        # time penalty
        """计算一步的各项奖励分量(只在 CONTINUE 时调用)。

        返回列表 [time_cost, rs_dist_reward, dist_reward, angle_reward, box_union_reward]:
            time_cost:        时间惩罚, -tanh(t / (10*TOLERANT_TIME)), 随时间变大;
            rs_dist_reward:   Reeds-Shepp 路径长度缩短带来的奖励(权重为 0 时跳过计算);
            dist_reward:      本步离终点的欧氏距离缩短量(按起终点距离归一化);
            angle_reward:     本步与目标朝向的夹角缩小量(按 π 归一化);
            box_union_reward: 车身与车位的 IoU 创新高时, 奖励新增的部分(只奖励"刷新纪录", 防止来回刷分)。
        最终奖励 = 这些分量按 configs.REWARD_WEIGHT 加权求和(见 env_wrapper.reward_shaping)。
        """
        time_cost = - np.tanh(self.t / (10*TOLERANT_TIME))

        # RS distance reward
        if REWARD_WEIGHT['rs_dist_reward'] != 0:
            radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
            curr_rs_dist = rsCurve.calc_optimal_path(*curr_state.get_pos(), *self.map.dest.get_pos(), radius , 0.1).L
            prev_rs_dist = rsCurve.calc_optimal_path(*prev_state.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_norm_ratio = rsCurve.calc_optimal_path(*self.map.start.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_reward = math.exp(-curr_rs_dist/rs_dist_norm_ratio) - \
                math.exp(-prev_rs_dist/rs_dist_norm_ratio)
        else:
            rs_dist_reward = 0

        # Euclidean distance reward & angle reward
        def get_angle_diff(angle1, angle2):
            # norm to 0 ~ pi/2
            """两个朝向的夹角, 归一化到 [0, π/2]。
            即车头朝前还是朝后停入都视为同样对齐。
            """
            angle_dif = math.acos(math.cos(angle1 - angle2)) # 0~pi
            return angle_dif if angle_dif<math.pi/2 else math.pi-angle_dif
        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = get_angle_diff(curr_state.heading, self.map.dest.heading)
        prev_dist_diff = prev_state.loc.distance(self.map.dest.loc)
        prev_angle_diff = get_angle_diff(prev_state.heading, self.map.dest.heading)
        dist_norm_ratio = max(self.map.dest.loc.distance(self.map.start.loc),10)
        angle_norm_ratio = math.pi
        dist_reward = prev_dist_diff/dist_norm_ratio - dist_diff/dist_norm_ratio
        angle_reward = prev_angle_diff/angle_norm_ratio - angle_diff/angle_norm_ratio
        
        # Box union reward
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        box_union_reward = union_area/(2*dest_box.area - union_area)
        if box_union_reward < self.accum_arrive_reward:
            box_union_reward = 0 
        else:
            prev_arrive_reward = self.accum_arrive_reward
            self.accum_arrive_reward = box_union_reward
            box_union_reward -= prev_arrive_reward
        return [time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward]
        
    def get_reward(self, status, prev_state):
        """status 为 CONTINUE 时返回 _get_reward 的分量, 否则返回全 0(终止奖励在 wrapper 中给出)。"""
        reward_info = [0,0,0,0,0]
        if status == Status.CONTINUE:
            reward_info = self._get_reward(prev_state, self.vehicle.state)
        return reward_info

    def step(self, action:np.ndarray = None):
        """环境推进一步。

        参数:
            action: [steer, speed] 真实物理量; 为 None 时不移动车辆, 只返回当前观测(reset 时使用)。

        流程:
            1. 把一个动作拆成 NUM_STEP(10) 个物理小步依次执行;
               每小步后若已到达则提前结束, 若碰撞则回退该小步并停止(车停在障碍物前);
            2. t += 1, 渲染并生成观测;
            3. 判定状态, 计算奖励分量;
            4. 如果离终点小于 RS_MAX_DIST(10 米), 尝试找一条无碰撞 RS 曲线, 放到 info['path_to_dest']。

        返回:
            observation (dict), reward_info (OrderedDict, 各奖励分量), status (Status), info (OrderedDict)。
        """
        assert self.vehicle is not None
        prev_state = self.vehicle.state
        collide = False
        arrive = False
        if action is not None:
            for simu_step_num in range(NUM_STEP):
                prev_info = self.vehicle.step(action,step_time=1)
                if self._check_arrived():
                    arrive = True
                    break
                if self._detect_collision():
                    if simu_step_num == 0:
                        collide = ENV_COLLIDE
                        self.vehicle.retreat(prev_info)
                    else:
                        self.vehicle.retreat(prev_info)
                    simu_step_num -= 1
                    break
            simu_step_num += 1
            # remove redundant trajectory
            if simu_step_num > 1:
                del self.vehicle.trajectory[-simu_step_num:-1]

        self.t += 1
        observation = self.render(self.render_mode)
        if arrive:
            status = Status.ARRIVED
        else:
            status = Status.COLLIDED if collide else self._check_status()

        reward_list = self.get_reward(status, prev_state)
        reward_info = OrderedDict({'time_cost':reward_list[0],\
            'rs_dist_reward':reward_list[1],\
            'dist_reward':reward_list[2],\
            'angle_reward':reward_list[3],\
            'box_union_reward':reward_list[4],})

        info = OrderedDict({'reward_info':reward_info,
            'path_to_dest':None})
        if self.t > 1 and status==Status.CONTINUE\
            and self.vehicle.state.loc.distance(self.map.dest.loc)<RS_MAX_DIST:
            rs_path_to_dest = self.find_rs_path(status)
            if rs_path_to_dest is not None:
                info['path_to_dest'] = rs_path_to_dest

        return observation, reward_info, status, info

    def _render(self, surface: pygame.Surface):
        """在 surface 上依次绘制: 背景、障碍物、起点框、终点车位、自车、最近 TRAJ_RENDER_LEN 步的轨迹。"""
        surface.fill(BG_COLOR)
        for obstacle in self.map.obstacles:
            pygame.draw.polygon(
                surface, OBSTACLE_COLOR, self._coord_transform(obstacle.shape))

        pygame.draw.polygon(
            surface, START_COLOR, self._coord_transform(self.map.start_box), width=1)
        pygame.draw.polygon(
            surface, DEST_COLOR, self._coord_transform(self.map.dest_box))
        
        pygame.draw.polygon(
            surface, self.vehicle.color, self._coord_transform(self.vehicle.box))

        if RENDER_TRAJ and len(self.vehicle.trajectory) > 1:
            render_len = min(len(self.vehicle.trajectory), TRAJ_RENDER_LEN)
            for i in range(render_len):
                vehicle_box = self.vehicle.trajectory[-(render_len-i)].create_box()
                pygame.draw.polygon(
                    surface, TRAJ_COLORS[-(render_len-i)], self._coord_transform(vehicle_box))

    def _get_img_observation(self, surface: pygame.Surface):
        """从全局画面中截取以自车为中心、按车头方向旋转后的 OBS_W x OBS_H 局部图像。

        返回:
            np.ndarray, 形状 (OBS_W, OBS_H, 3), uint8 RGB。
        """
        angle = self.vehicle.state.heading
        old_center = surface.get_rect().center

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(surface, np.rad2deg(angle))
        rotate = pygame.Surface((WIN_W, WIN_H))
        rotate.blit(capture, capture.get_rect(center=old_center))
        
        vehicle_center = np.array(self._coord_transform(self.vehicle.box.centroid)[0])
        dx = (vehicle_center[0]-old_center[0])*np.cos(angle) \
            + (vehicle_center[1]-old_center[1])*np.sin(angle)
        dy = -(vehicle_center[0]-old_center[0])*np.sin(angle) \
            + (vehicle_center[1]-old_center[1])*np.cos(angle)
        
        # align the center of the observation with the center of the vehicle
        observation = pygame.Surface((WIN_W, WIN_H))
    
        observation.fill(BG_COLOR)
        observation.blit(rotate, (int(-dx), int(-dy)))
        observation = observation.subsurface((
            (WIN_W-OBS_W)/2, (WIN_H-OBS_H)/2), (OBS_W, OBS_H))

    
        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.raw_img_shape)

        return observation

    def _process_img_observation(self, img):
        """把原始 RGB 图像处理为网络输入(背景变黑、4 倍降采样、归一化到 [0,1])。

        参数:
            img: 形状 (OBS_W, OBS_H, 3)。
        返回:
            形状 (OBS_W//4, OBS_H//4, 3)。
        """
        processed_img = self.img_processor.process_img(img)
        return processed_img

    def _get_lidar_observation(self,):
        """调用激光雷达仿真器, 返回 (LIDAR_NUM,) 的距离数组。"""
        obs_list = [obs.shape for obs in self.map.obstacles]
        lidar_view = self.lidar.get_observation(self.vehicle.state, obs_list)
        return lidar_view
    
    def _get_targt_repr(self,):
        # target position representation
        """计算目标车位相对自车的表示, 长度 5:
            [相对距离, cos(目标方位角), sin(目标方位角), cos(相对朝向), cos(相对朝向)]。
        目标方位角 = 自车指向终点的连线方向 - 自车朝向; 相对朝向 = 终点朝向 - 自车朝向。
        注意: 最后一项原注释的意图是 sin(相对朝向), 代码里写成了 cos。
        """
        dest_pos = (self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading)
        ego_pos = (self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle),\
            math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr 

    def render(self, mode: str = "human"):
        """渲染当前画面并生成观测 dict。

        参数:
            mode: "human" 显示窗口, "rgb_array" 隐藏窗口。
        返回:
            {'img', 'lidar', 'target', 'action_mask'}, 未启用的模态为 None。
        即使不需要显示, 训练时也必须调用它, 因为图像观测来自 pygame 渲染结果。
        """
        assert mode in self.metadata["render_mode"]
        assert self.vehicle is not None

        if mode == "human":
            display_flags = pygame.SHOWN
        else:
            display_flags = pygame.HIDDEN
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags = display_flags)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._render(self.screen)
        observation = {'img':None, 'lidar':None, 'target':None, 'action_mask':None}
        if self.use_img_observation:
            raw_observation = self._get_img_observation(self.screen)
            observation['img'] = self._process_img_observation(raw_observation)
        if self.use_lidar_observation:
            observation['lidar'] = self._get_lidar_observation()
        if self.use_action_mask:
            observation['action_mask'] = self.action_filter.get_steps(observation['lidar'])
        observation['target'] = self._get_targt_repr()
        pygame.display.update()
        self.clock.tick(self.fps)
        
        return observation

    def find_rs_path(self,status):
        """从当前位置到终点搜索一条无碰撞的 Reeds-Shepp 曲线。

        做法: 计算所有 RS 候选路径, 按长度从短到长依次做碰撞检测,
        返回第一条无碰撞的路径; 长度超过最短路径 1.6 倍(且已检查超过 2 条)时放弃。

        返回:
            reeds_shepp.PATH 对象, 或 None(找不到)。
        """
        startX, startY, startYaw = self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading
        goalX, goalY, goalYaw = self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading
        radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
        #  Find all possible reeds-shepp paths between current and goal node
        reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 0.1)

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths:
            return None

        # Find path with lowest cost considering non-holonomic constraints
        costQueue = heapdict()
        for path in reedsSheppPaths:
            costQueue[path] = path.L

        # Find first path in priority queue that is collision free
        min_path_len = -1
        idx = 0
        while len(costQueue)!=0:
            idx += 1
            path = costQueue.popitem()[0]
            if min_path_len < 0:
                min_path_len = path.L
            if path.L > 1.6*min_path_len and idx > 2:
                break
            traj=[]
            traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
            traj_valid = self.is_traj_valid(traj)
            if traj_valid:
                return path
        return None
    
    def is_traj_valid(self, traj):
        """检查一条轨迹是否无碰撞且不出界。

        参数:
            traj: [[x, y, yaw], ...] 轨迹点。
        做法: 在每个轨迹点放置车身矩形, 用向量化的"线段求交"判断车身边是否与附近障碍物的边相交。
        返回:
            True 表示可行(不出界、无相交)。
        """
        car_coords1 = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords2 = np.array(VehicleBox.coords)[1:] # (4,2)
        car_coords_x1 = car_coords1[:,0].reshape(1,-1)
        car_coords_y1 = car_coords1[:,1].reshape(1,-1) # (1,4)
        car_coords_x2 = car_coords2[:,0].reshape(1,-1)
        car_coords_y2 = car_coords2[:,1].reshape(1,-1) # (1,4)
        vxs = np.array([t[0] for t in traj])
        vys = np.array([t[1] for t in traj])
        # check outbound
        if np.min(vxs) < self.map.xmin or np.max(vxs) > self.map.xmax \
        or np.min(vys) < self.map.ymin or np.max(vys) > self.map.ymax:
            return False
        vthetas = np.array([t[2] for t in traj])
        cos_theta = np.cos(vthetas).reshape(-1,1) # (T,1)
        sin_theta = np.sin(vthetas).reshape(-1,1)
        vehicle_coords_x1 = cos_theta*car_coords_x1 - sin_theta*car_coords_y1 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y1 = sin_theta*car_coords_x1 + cos_theta*car_coords_y1 + vys.reshape(-1,1)
        vehicle_coords_x2 = cos_theta*car_coords_x2 - sin_theta*car_coords_y2 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y2 = sin_theta*car_coords_x2 + cos_theta*car_coords_y2 + vys.reshape(-1,1)
        vx1s = vehicle_coords_x1.reshape(-1,1)
        vx2s = vehicle_coords_x2.reshape(-1,1)
        vy1s = vehicle_coords_y1.reshape(-1,1)
        vy2s = vehicle_coords_y2.reshape(-1,1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (vy2s - vy1s).reshape(-1,1) # (4*t,1)
        b = (vx1s - vx2s).reshape(-1,1)
        c = (vy1s*vx2s - vx1s*vy2s).reshape(-1,1)
        
        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x_max = np.max(vx1s) + 5
        x_min = np.min(vx1s) - 5
        y_max = np.max(vy1s) + 5
        y_min = np.min(vy1s) - 5

        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in self.map.obstacles:
            if isinstance(obst, Area):
                obst = obst.shape
            obst_coords = np.array(obst.coords) # (n+1,2)
            if (obst_coords[:,0] > x_max).all() or (obst_coords[:,0] < x_min).all()\
                or (obst_coords[:,1] > y_max).all() or (obst_coords[:,1] < y_min).all():
                continue
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return True
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # calculate the intersections
        det = a*e - b*d # (4, E)
        parallel_line_pos = (det==0) # (4, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (4, E)
        raw_y = (c*d - a*f)/det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        collide_map_x[raw_x>np.maximum(x1s, x2s)] = 0
        collide_map_x[raw_x<np.minimum(x1s, x2s)] = 0
        collide_map_y[raw_y>np.maximum(y1s, y2s)] = 0
        collide_map_y[raw_y<np.minimum(y1s, y2s)] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x>np.maximum(vx1s, vx2s)] = 0
        collide_map_x[raw_x<np.minimum(vx1s, vx2s)] = 0
        collide_map_y[raw_y>np.maximum(vy1s, vy2s)] = 0
        collide_map_y[raw_y<np.minimum(vy1s, vy2s)] = 0

        collide_map = collide_map_x*collide_map_y
        collide_map[parallel_line_pos] = 0
        collide = np.sum(collide_map) > 0

        if collide:
            return False
        return True

    def close(self):
        """关闭 pygame 窗口。"""
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()

