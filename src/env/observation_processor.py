"""图像观测的预处理: 背景变黑、降采样、归一化。"""
import cv2
import numpy as np

from configs import *
    
class Obs_Processor():
    """把渲染得到的 RGB 鸟瞰图处理成神经网络的图像输入。"""
    def __init__(self) -> None:
        """downsample_rate=4: 256x256 -> 64x64; n_channels=3 (RGB)。"""
        self.downsample_rate = 4
        self.n_channels = 3

    def process_img(self, img):
        """处理一帧图像。

        参数:
            img: 原始 RGB 图像, 形状 (OBS_W, OBS_H, 3), uint8。
        返回:
            形状 (OBS_W/4, OBS_H/4, 3) 的 float 图像, 数值范围 [0, 1], 背景为黑色(0)。
        """
        processed_img = self.change_bg_color(img)
        processed_img = cv2.resize(processed_img, (img.shape[0]//self.downsample_rate, img.shape[1]//self.downsample_rate))
        processed_img = processed_img/255.0

        return processed_img

    def change_bg_color(self, img):
        """把与 BG_COLOR(白色) 完全相同的像素改成黑色, 使"空地"在网络输入里为 0。"""
        processed_img = img.copy()
        bg_pos = img==BG_COLOR[:3]
        bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        processed_img[bg_pos] = (0,0,0)
        return processed_img
    
