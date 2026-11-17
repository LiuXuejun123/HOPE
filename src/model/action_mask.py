"""动作掩码(Action Mask), HOPE 的关键组件之一。

把连续动作空间离散成 42 个动作(21 个转角 x 前进/倒车), 根据当前激光雷达观测,
计算每个离散动作"在不碰撞的前提下最多能走完多少比例", 得到一个 42 维、取值 [0,1] 的向量:
    1   表示这个动作完整走完也安全;
    0   表示一动就会撞;
    0.3 表示只能安全走完 30%。
这个向量既作为观测输入给网络, 也(在 PPO 中)用来对动作采样概率加权, 从而把探索限制在安全动作上。

加速技巧: 每个动作的扫过区域只与车辆几何有关, 与环境无关, 所以在初始化时预先算好 dist_star,
运行时只需和激光雷达读数做比较。
"""
import numpy as np
import torch
from shapely.geometry import LineString, Point
from scipy.ndimage.filters import minimum_filter1d

from configs import *

class ActionMask():
    """动作掩码计算器。主要接口: get_steps(lidar) 计算掩码; choose_action(mean, std, mask) 按掩码采样动作。"""
    def __init__(self, VehicleBox=VehicleBox, n_iter=10) -> None:
        """参数:
            VehicleBox: 车身模板(以后轴中心为原点)。
            n_iter:     把每个离散动作拆成多少个小步来检测碰撞。
        初始化时预计算每个动作每一小步的车身位置 vehicle_boxes 和距离表 dist_star。
        """
        print('initializing action mask')
        self.vehicle_box_base = VehicleBox
        self.n_iter = n_iter
        self.action_space = discrete_actions
        self.vehicle_boxes = self.init_vehicle_box() # (42, n_iter, 4, 2)
        self.lidar_num = LIDAR_NUM
        self.lidar_range = LIDAR_RANGE
        self.vehicle_lidar_base = self.get_vehicle_lidar_base()
        self.up_sample_rate = 10
        self.dist_star = self.precompute()

    def get_vehicle_lidar_base(self, ):
        """计算每条雷达射线上"原点到车身边缘"的距离, 形状 (lidar_num,)。"""
        lidar_base = []
        ORIGIN = Point((0,0))
        for l in range(self.lidar_num):
            lidar_line = LineString(((0,0), (np.cos(l*np.pi/self.lidar_num*2)*self.lidar_range,\
                 np.sin(l*np.pi/self.lidar_num*2)*self.lidar_range)))
            distance = lidar_line.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def _intersect(self, e1:np.ndarray, e2:np.ndarray):
        """批量计算两组线段的交点。

        参数:
            e1: m 条线段, 形状 (m, 2, 2)。
            e2: n 条线段, 形状 (n, 2, 2)。
        返回:
            形状 (m, n, 2) 的交点坐标; 不相交的位置为 +inf。
        """
        # calculate the params for edges 1
        x1s1, x2s1, y1s1, y2s1 = e1[:, 0, 0].reshape(-1,1), e1[:, 1, 0].reshape(-1,1), e1[:, 0, 1].reshape(-1,1), e1[:, 1, 1].reshape(-1,1)
        a = (y2s1 - y1s1).reshape(-1,1) # (m,1)
        b = (x1s1 - x2s1).reshape(-1,1)
        c = (y1s1*x2s1 - x1s1*y2s1).reshape(-1,1)

        # calculate the params for edges 2
        x1s2, x2s2, y1s2, y2s2 = e2[:, 0, 0].reshape(1,-1), e2[:, 1, 0].reshape(1,-1), e2[:, 0, 1].reshape(1,-1), e2[:, 1, 1].reshape(1,-1)
        d = (y2s2 - y1s2).reshape(1,-1) # (1,n)
        e = (x1s2 - x2s2).reshape(1,-1)
        f = (y1s2*x2s2 - x1s2*y2s2).reshape(1,-1)

        # calculate the intersections by line(instead of edge) 1 & 2
        det = a*e - b*d # (m, n)
        parallel_line_pos = (det==0) # (m, n)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (m, n)
        raw_y = (c*d - a*f)/det

        # select the true intersections, set the false positive interesections to inf
        tmp_inf = np.inf
        tolerrance = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        raw_x[raw_x>np.maximum(x1s1, x2s1)+tolerrance] = tmp_inf
        raw_x[raw_x<np.minimum(x1s1, x2s1)-tolerrance] = tmp_inf
        raw_y[raw_y>np.maximum(y1s1, y2s1)+tolerrance] = tmp_inf
        raw_y[raw_y<np.minimum(y1s1, y2s1)-tolerrance] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x>np.maximum(x1s2, x2s2)+tolerrance] = tmp_inf
        raw_x[raw_x<np.minimum(x1s2, x2s2)-tolerrance] = tmp_inf
        raw_y[raw_y>np.maximum(y1s2, y2s2)+tolerrance] = tmp_inf
        raw_y[raw_y<np.minimum(y1s2, y2s2)-tolerrance] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        intersections = np.stack([raw_x, raw_y], axis=2) # (m, n, 2)
        # print(e1.shape, e2.shape, intersections.shape)
        assert intersections.shape ==  (e1.shape[0], e2.shape[0], 2)

        return intersections
    
    def init_vehicle_box(self,):
        """预先模拟: 车从原点出发, 对每个离散动作沿圆弧(由转角决定半径)前进 n_iter 个小步,
        记录每一小步的车身四个角点。

        返回:
            形状 (n_action, n_iter, 4, 2) 的数组。
        """
        VehicleBox = self.vehicle_box_base
        vehicle_boxes = []
        x,y,theta = 0,0,0
        actions = np.array(self.action_space)
        radius = 1/(np.tan(actions[:,0])/WHEEL_BASE)
        car_coords = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords_x = car_coords[:,0].reshape(1,-1)
        car_coords_y = car_coords[:,1].reshape(1,-1) # (1,4)
        Ox = x-radius*np.sin(theta)
        Oy = y+radius*np.cos(theta)
        delta_phi = 0.5*actions[:,1]/10/radius # (42)
        ptheta = theta
        px, py = x,y

        for _ in range(self.n_iter):
            ptheta = ptheta + delta_phi # (42)
            px = Ox + radius*np.sin(ptheta) # (42)
            py = Oy - radius*np.cos(ptheta) # (42)

            # coords transform
            cos_theta = np.cos(ptheta).reshape(-1,1) # (42)
            sin_theta = np.sin(ptheta).reshape(-1,1)
            vehicle_coords_x = cos_theta*car_coords_x - sin_theta*car_coords_y + px.reshape(-1,1) # (42,4)
            vehicle_coords_y = sin_theta*car_coords_x + cos_theta*car_coords_y + py.reshape(-1,1) # (42,4)
            vehicle_coords = np.concatenate((np.expand_dims(vehicle_coords_x, axis=-1), np.expand_dims(vehicle_coords_y, axis=-1)), axis=-1) # (42,4,2)
            vehicle_boxes.append(vehicle_coords)
        
        return np.array(vehicle_boxes).transpose(1,0,2,3)

    def precompute(self,):
        """预计算 dist_star, 用于快速求动作掩码。

        返回:
            dist_star, 形状 (lidar_num * up_sample_rate, n_action, n_iter)。
            dist_star[i, j, k] 表示: 执行第 j 个动作走到第 k 小步时, 车身沿第 i 条射线方向伸出的最远距离。
            只要该方向上的障碍物比这个距离更远, 这一小步在该方向就不会碰撞。
        射线方向通过线性插值上采样 10 倍, 使检测更密集。
        """
        lidar_line_idx = np.arange(self.lidar_num)
        max_distance = LIDAR_RANGE*10
        lidar_line_end_x = np.cos(lidar_line_idx/self.lidar_num*2*np.pi)*max_distance
        lidar_line_end_y = np.sin(lidar_line_idx/self.lidar_num*2*np.pi)*max_distance
        lidar_line_end_coords = np.stack([lidar_line_end_x, lidar_line_end_y], axis=1) # (lidar_num, 2)
        origin = np.zeros_like(lidar_line_end_coords)
        lidar_edges = np.stack([origin, lidar_line_end_coords], axis=1) # (lidar_num, 2, 2)

        vehicle_boxes_shifed = np.zeros_like(self.vehicle_boxes)
        vehicle_boxes_shifed[:, :, :-1, :] = self.vehicle_boxes[:, :, 1:, :]
        vehicle_boxes_shifed[:, :, -1, :] = self.vehicle_boxes[:, :, 0, :]
        vehicle_edges = np.stack([vehicle_boxes_shifed, self.vehicle_boxes],axis=3) # (n_action, n_iter, 4, 2, 2)
        vehicle_edges = vehicle_edges.reshape(-1, 2, 2) # (n_action*n_iter*4, 2, 2)
        intersections = self._intersect(lidar_edges, vehicle_edges) # (lidar_num, n_action*n_iter*4, 2)
        # intersections[intersections==np.inf] = 0
        intersections = intersections.reshape(self.lidar_num, len(self.action_space), self.n_iter, 4, 2)
        intersections = np.linalg.norm(intersections, axis=-1)
        intersections[intersections==np.inf] = 0
        dist_star = np.max(intersections, axis=-1) # (lidar_num, n_action, n_iter)
        dist_star = self._linear_interpolate(dist_star, self.up_sample_rate)
        return dist_star
    
    def _linear_interpolate(self, x:np.ndarray, upsample_rate:int=None):
        """沿第 0 维做线性插值上采样(把首尾视为相连的环形, 适用于 360 度雷达)。

        参数:
            x: 输入数组。
            upsample_rate: 上采样倍数。
        返回:
            第 0 维长度变为 x.shape[0] * upsample_rate 的数组。
        """
        if upsample_rate is None:
            upsample_rate = self.up_sample_rate
        y = np.zeros((x.shape[0]*upsample_rate,) + x.shape[1:])
        x = np.concatenate([x, x[0:1]], axis=0) # assume the input is circular
        j = np.arange(y.shape[0])
        tmp_shape = (y.shape[0],) + (1,)*(len(x.shape)-1)
        y = x[j//upsample_rate,...]*(1-(j%upsample_rate)/upsample_rate).reshape(tmp_shape) +\
            x[(j)//upsample_rate+1,...]*(((j)%upsample_rate)/upsample_rate).reshape(tmp_shape)
        return y


    def get_steps(self, raw_lidar_obs:np.ndarray):
        """根据当前激光雷达观测计算动作掩码。

        参数:
            raw_lidar_obs: 环境给出的雷达观测(车身到障碍物的距离, 形状 (lidar_num,))。
        返回:
            形状 (n_action,) 的数组, 每个值 ∈ [0, 1], 表示该动作可安全执行的比例。
            若全部为 0, 则统一设为 0.01, 避免后续按掩码采样时概率全为 0。

        步骤: 把雷达读数换回"原点到障碍物距离"并上采样 -> 与 dist_star 比较得到每小步是否安全
        -> 对每个动作找第一个不安全的小步 -> 所有射线取最小值 -> post_process 平滑。
        """
        lidar_obs = np.clip(raw_lidar_obs, 0, 10) + self.vehicle_lidar_base
        dist_obs = self._linear_interpolate(lidar_obs.reshape(-1), self.up_sample_rate).reshape(-1,1,1) # (lidar_num*upsample_rate, 1, 1)
        step_save = np.zeros_like(self.dist_star) # (lidar_num, n_action, n_iter)
        step_save[self.dist_star<=dist_obs] = 1
        step_save[self.dist_star>dist_obs] = 0
        # find the first "0" on the last axis
        max_step = np.argmin(step_save, axis=-1) # (lidar_num, n_action)
        max_step[np.sum(step_save, axis=-1) == self.n_iter] = self.n_iter
        
        step_len = np.min(max_step, axis=0) # (n_action)

        step_len = self.post_process(step_len)
        if np.sum(step_len) == 0:
            return np.clip(step_len, 0.01, 1)
        return step_len
    
    def post_process(self, step_len:np.ndarray):
        """对每个动作的可行步数做保守化处理:
        把最左/最右转角的步数各减 1, 再在相邻转角间做宽度为 5 的最小值滤波(邻近动作不安全时自己也变保守),
        最后除以 n_iter 归一化到 [0, 1]。前进和倒车两组分别处理。
        """
        kernel = 5
        forward_step_len = step_len[:len(step_len)//2]
        backward_step_len = step_len[len(step_len)//2:]
        forward_step_len[0] -= 1
        forward_step_len[-1] -= 1
        backward_step_len[0] -= 1
        backward_step_len[-1] -= 1
        forward_step_len_ = minimum_filter1d(forward_step_len, kernel)
        backward_step_len_ = minimum_filter1d(backward_step_len, kernel)
        return np.clip(np.concatenate((forward_step_len_, backward_step_len_)), 0, self.n_iter)/self.n_iter
    
    
    def choose_action(self, action_mean, action_std, action_mask):

        """结合策略网络输出的高斯分布和动作掩码, 从 42 个离散动作中采样一个。

        参数:
            action_mean, action_std: 策略网络给出的高斯分布均值和标准差(归一化动作空间)。
            action_mask:             get_steps 给出的掩码。
        返回:
            选中的动作, 已归一化到 [-1, 1](转角除以最大转角, 速度为 ±1)。

        做法: 对每个离散动作计算高斯对数概率 -> 取 exp 后乘以掩码 -> 归一化成分布 -> 随机采样。
        """
        if isinstance(action_mean, torch.Tensor):
            action_mean = action_mean.cpu().numpy()
            action_std = action_std.cpu().numpy()
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.cpu().numpy()
        if len(action_mean.shape) == 2:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        if len(action_mask.shape) == 2:
            action_mask = action_mask.squeeze(0)

        def calculate_probability(mean, std, values):
            """计算 values 中每个动作在给定高斯分布下的对数概率(各维度裁剪到 [-10, 10] 后求和)。"""
            z_scores = (values - mean) / std
            log_probabilities = -0.5 * z_scores ** 2 - np.log((np.sqrt(2 * np.pi) * std))
            return np.sum(np.clip(log_probabilities, -10, 10), axis=1)

        possible_actions = np.array(self.action_space)
        # deal the scaling
        scale_steer = VALID_STEER[1]
        scale_speed = 1
        possible_actions = possible_actions/np.array([scale_steer, scale_speed])
        prob = calculate_probability(action_mean, action_std, possible_actions)
        exp_prob = np.exp(prob) * action_mask
        prob_softmax = exp_prob / np.sum(exp_prob)
        actions = np.arange(len(possible_actions))
        action_chosen = np.random.choice(actions, p=prob_softmax)
        return possible_actions[action_chosen]
