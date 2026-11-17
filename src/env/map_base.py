"""地图元素的基础数据结构。"""
import numpy as np
from shapely.geometry.base import BaseGeometry

class Area(object):
    """地图上的一个区域(目前只用于表示障碍物)。

    属性:
        shape:   shapely 几何体(通常是 LinearRing)。
        subtype: 类型字符串, 例如 "obstacle"。
        color:   渲染颜色 RGBA。
    """
    def __init__(
        self,
        shape: BaseGeometry = None, 
        subtype: str = None,
        color: float = None,
    ):
        """保存形状、类型和颜色。"""
        self.shape = shape
        self.subtype = subtype
        self.color = color

    def get_shape(self):
        """返回形状的坐标数组, 形状 (n_points, 2)。"""
        return np.array(self.shape.coords)
