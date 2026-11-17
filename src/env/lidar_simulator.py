"""二维激光雷达仿真。

以后轴中心为原点, 均匀发出 LIDAR_NUM(120) 条射线, 计算每条射线碰到障碍物的距离。
输出的是"车身外缘到障碍物的距离"(已减去原点到车身边缘的距离)。
"""
import sys
sys.path.append("../")
sys.path.append(".")
import math
import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform

from env.vehicle import State,VehicleBox

ORIGIN = Point((0,0))

class LidarSimlator():
    """激光雷达仿真器。第 i 条射线的方向是 (车头方向 + i * 2π / lidar_num), 第 0 条朝正前方。"""
    def __init__(self, 
        lidar_range:float = 10.0,
        lidar_num:int = 120
    ) -> None:
        """参数:
            lidar_range: 最大探测距离(米), 超过这个距离视为没有障碍物。
            lidar_num:   射线条数。
        初始化时预先生成所有射线, 并算好每条射线上"原点到车身边缘"的距离 vehicle_boundary。
        """
        self.lidar_range = lidar_range
        self.lidar_num = lidar_num
        self.lidar_lines = []
        for a in range(lidar_num):
            self.lidar_lines.append(LineString(((0,0), (math.cos(a*math.pi/lidar_num*2)*lidar_range,\
                 math.sin(a*math.pi/lidar_num*2)*lidar_range))))
        self.vehicle_boundary = self.get_vehicle_boundary()

    def get_observation(self, ego_state:State, obstacles:list):
        """从自车视角获取一帧激光雷达观测。

        参数:
            ego_state: 自车状态。
            obstacles: 障碍物列表(LinearRing)。

        返回:
            np.ndarray, 形状 (lidar_num,)。每个值 = 射线命中距离 - 车身边缘距离,
            即"车身表面沿该方向到最近障碍物的距离", 无障碍时约为 lidar_range 减去车身边缘距离。
        """

        ego_pos = (ego_state.loc.x, ego_state.loc.y, ego_state.heading)
        rotated_obstacles = self._rotate_and_filter_obstacles(ego_pos, obstacles)
        lidar_obs = self._fast_calc_lidar_obs(rotated_obstacles)
        return np.array(lidar_obs - self.vehicle_boundary)
    
    def get_vehicle_boundary(self, ):
        """计算每条射线从原点(后轴中心)到车身矩形边缘的距离, 形状 (lidar_num,)。
        用于把"原点到障碍物距离"换算成"车身到障碍物距离"。
        """
        lidar_base = []
        for l in self.lidar_lines:
            distance = l.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def _rotate_and_filter_obstacles(self, ego_pos:tuple, obstacles:list):
        """把障碍物从世界坐标变换到自车坐标系(自车在原点、车头朝 x 正方向),
        并丢弃距离超过 lidar_range 的障碍物以减少计算量。

        参数:
            ego_pos: (x, y, heading)。
            obstacles: 世界坐标下的障碍物列表。
        返回:
            自车坐标系下、在探测范围内的障碍物列表。
        """
        x, y, theta = ego_pos
        a = math.cos(theta)
        b = math.sin(theta)
        x_off = -x*a - y*b
        y_off = x*b - y*a
        affine_mat = [a, b, -b, a, x_off, y_off]

        rotated_obstacles = []
        for obs in obstacles:
            rotated_obs = affine_transform(obs, affine_mat)
            if rotated_obs.distance(ORIGIN) < self.lidar_range:
                rotated_obstacles.append(rotated_obs)
        
        return rotated_obstacles
    
    def _fast_calc_lidar_obs(self, obstacles:list):
        """用 numpy 矩阵运算一次性求出所有射线与所有障碍物边的交点, 取每条射线的最近交点距离。

        做法: 把射线和障碍物的每条边都写成直线方程 ax+by+c=0, 批量求交点,
        再把"不在射线正方向上"或"不在线段范围内"的假交点设为一个大数, 最后按行取最小值。

        参数:
            obstacles: 已变换到自车坐标系的障碍物(LinearRing)列表。
        返回:
            np.ndarray, 形状 (lidar_num,), 每条射线从原点出发到最近障碍物的距离, 裁剪到 [0, lidar_range]。
        """

        # Line 1: the lidar ray, ax + by + c = 0
        theta = np.array([a*math.pi/self.lidar_num*2 for a in range(self.lidar_num)]) # (120,)
        a = np.sin(theta).reshape(-1,1) # (120, 1)
        b = -np.cos(theta).reshape(-1,1)
        c = 0

        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in obstacles:
            obst_coords = np.array(obst.coords) # (n+1,2)
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return np.ones((self.lidar_num))*self.lidar_range
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # calculate the intersections
        det = a*e - b*d # (120, E)
        parallel_line_pos = (det==0) # (120, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (120, E)
        raw_y = (c*d - a*f)/det

        # select the true intersections, set the false positive interesections to inf
        tmp_inf = 100
        tmp_zero = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        # here we assume the orientation of lidar[0] is 0 rad in the ego coordinate.
        raw_x[:self.lidar_num//4][raw_x[:self.lidar_num//4]<-tmp_zero] = tmp_inf
        raw_x[self.lidar_num//4*3:][raw_x[self.lidar_num//4*3:]<-tmp_zero] = tmp_inf
        raw_x[self.lidar_num//4:self.lidar_num//4*3][raw_x[self.lidar_num//4:self.lidar_num//4*3]>tmp_zero] = tmp_inf
        raw_y[:self.lidar_num//2][raw_y[:self.lidar_num//2]<-tmp_zero] = tmp_inf
        raw_y[self.lidar_num//2:][raw_y[self.lidar_num//2:]>tmp_zero] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x>np.maximum(x1s, x2s)] = tmp_inf
        raw_x[raw_x<np.minimum(x1s, x2s)] = tmp_inf
        raw_y[raw_y>np.maximum(y1s, y2s)] = tmp_inf
        raw_y[raw_y<np.minimum(y1s, y2s)] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        lidar_obs = np.min(np.sqrt(raw_x**2 + raw_y**2), axis=1) # (120,)
        lidar_obs = np.clip(lidar_obs, 0, self.lidar_range)
        return lidar_obs


