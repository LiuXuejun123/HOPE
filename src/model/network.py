"""神经网络结构。

核心是 MultiObsEmbedding: actor(策略网络)和 critic(价值网络)都用它, 只是输出维度不同。
其余为图像编码器(ImgEncoder)以及用于预训练图像编码器的自编码器(AE_Conv / VAE_Conv)。
"""
from typing import List, OrderedDict

import torch
import torch.nn as nn
from torch import cat

from model.attention import AttentionNetwork

def orthogonal_init(layer, gain=1.0):
    """对一个线性层做正交初始化, 偏置置 0(当前未被调用)。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Network(nn.Module):
    """通用 MLP: 由 (名字, 层) 列表构成的 nn.Sequential(当前未被使用)。"""
    def __init__(self, layers: list, orthogonal_init: bool = True):
        """layers: [(name, nn.Module), ...]; orthogonal_init: 是否做正交初始化。"""
        super().__init__()
        self.net = nn.Sequential(OrderedDict(layers))
        if orthogonal_init:
            self.orthogonal_init()

    def orthogonal_init(self):
        """对所有权重做正交初始化、偏置置 0。"""
        i = 0
        for layer_name, layer in self.net.state_dict().items():
            # The output layer is specially dealt
            gain = 1 if i < len(self.net.state_dict()) - 2 else 0.01
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

    def forward(self, x):
        """前向传播。"""
        out = self.net(x)
        return out

class MultiObsEmbedding(nn.Module):
    """多模态观测编码网络(actor 和 critic 共用的结构)。

    结构:
        lidar (120)        -> MLP -> 128 维 token
        target (5)         -> MLP -> 128 维 token
        action_mask (42)   -> MLP -> 128 维 token
        img (3x64x64)      -> 预训练 CNN 编码器 -> 线性层 -> 128 维 token
        [action (2)]       -> MLP -> 128 维 token(仅 SAC 的 critic 使用)
        所有 token 堆叠为 (batch, n_modal, 128) -> Transformer -> 展平 -> MLP -> 输出
        actor 输出 2 维(经 tanh 到 [-1,1], 作为高斯分布均值), critic 输出 1 维(状态价值)。
    不使用注意力时, 改为把各 token 拼接后接普通 MLP。
    """
    def __init__(self, configs):
        """参数:
            configs: 网络配置 dict, 见 configs.py 中的 ACTOR_CONFIGS / CRITIC_CONFIGS。
        """
        super().__init__()
        embed_size = configs['embed_size']
        hidden_size = configs['hidden_size']
        activate_func = [nn.LeakyReLU(), nn.Tanh()][configs['use_tanh_activate']]
        self.use_img = False if configs['img_shape'] is None else True
        self.use_action_mask = False if configs['action_mask_shape'] is None else True
        self.use_attention = False if configs['attention_configs'] is None else True
        self.input_action = 'input_action_dim' in configs and configs['input_action_dim'] > 0

        if not self.use_attention:
            if configs['n_hidden_layers'] == 1:
                layers = [nn.Linear(configs['n_modal']*embed_size, configs['output_size'])]
            else:
                layers = [nn.Linear(configs['n_modal']*embed_size, hidden_size)]
                for _ in range(configs['n_hidden_layers']-2):
                    layers.append(activate_func)
                    layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Linear(hidden_size, configs['output_size']))
            self.net = nn.Sequential(*layers)
        else:
            attention_configs = configs['attention_configs']
            self.net = AttentionNetwork(
                embed_size,
                attention_configs['depth'],
                attention_configs['heads'],
                attention_configs['dim_head'],
                attention_configs['mlp_dim'],
                configs['n_modal'],
                attention_configs['hidden_dim'],
                configs['output_size'],
            )
        self.output_layer = nn.Tanh() if configs['use_tanh_output'] else None

        if configs['lidar_shape'] is not None:
            layers = [nn.Linear(configs['lidar_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_lidar = nn.Sequential(*layers)

        if configs['target_shape'] is not None:
            layers = [nn.Linear(configs['target_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_tgt = nn.Sequential(*layers)
            
        if configs['action_mask_shape'] is not None:
            layers = [nn.Linear(configs['action_mask_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_am = nn.Sequential(*layers)

        if configs['img_shape'] is not None:
            self.embed_img = ImgEncoder(configs['img_shape'], configs['k_img_conv'],\
                                    embed_size, configs['img_conv_layers'], configs['img_linear_layers'])
            self.re_embed_img = nn.Sequential(activate_func, nn.Linear(embed_size, embed_size)) # the latten vector may not be scaled

        if self.input_action:
            layers = [nn.Linear(configs['input_action_dim'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_action = nn.Sequential(*layers)

        if orthogonal_init:
            self.orthogonal_init()

    def orthogonal_init(self):
        """对融合网络和各模态嵌入层做正交初始化、偏置置 0(PPO 常用的初始化技巧)。"""
        i = 0
        for layer_name, layer in self.net.state_dict().items():
            # The output layer is specially dealt
            gain = 1 if i < len(self.net.state_dict()) - 2 else 0.01
            if layer_name.endswith("weight") and len(layer.shape)>1:
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        for layer_name, layer in self.embed_lidar.state_dict().items():
            # The output layer is specially dealt
            gain = 1
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        for layer_name, layer in self.embed_tgt.state_dict().items():
            # The output layer is specially dealt
            gain = 1
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

        if self.use_action_mask:
            for layer_name, layer in self.embed_am.state_dict().items():
                # The output layer is specially dealt
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)
        
        if self.use_img:
            for layer_name, layer in self.re_embed_img.state_dict().items():
                # The output layer is specially dealt
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)
        
        if self.input_action:
            for layer_name, layer in self.embed_action.state_dict().items():
                # The output layer is specially dealt
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)

    def load_img_encoder(self, path, device, require_grad = False):
        """用预训练自编码器的编码器替换图像编码部分。

        参数:
            path:         自编码器模型文件(model/ckpt/autoencoder.pt)。
            device:       加载到的设备。
            require_grad: 是否继续训练图像编码器(默认冻结)。
        """
        ae = torch.load(path, map_location=device)
        self.embed_img = ae.encoder
        for param in self.embed_img.parameters():
            param.requires_grad = require_grad

    def forward(self, x:dict):
        """参数:
            x: 观测 dict, 每个值是带 batch 维的张量:
               'lidar' (B, 120), 'target' (B, 5), 'action_mask' (B, 42), 'img' (B, 3, 64, 64), [可选 'action' (B, 2)]。
        返回:
            (B, output_size) 张量。
        """
        feature_lidar = self.embed_lidar(x['lidar'])
        feature_target = self.embed_tgt(x['target'])
        features = [feature_lidar, feature_target]
        if self.use_action_mask:
            feature_am = self.embed_am(x['action_mask'])
            features.append(feature_am)

        if self.use_img:
            feature_img, _ = self.embed_img(x['img'])
            feature_img = self.re_embed_img(feature_img)
            features.append(feature_img)

        if self.input_action:
            feature_action = self.embed_action(x['action'])
            features.append(feature_action)

        if self.use_attention:
            embed = torch.stack(features, dim=1)
        else:
            embed = cat(features, dim=1)
        out = self.net(embed)
        if self.output_layer is not None:
            out = self.output_layer(out)
        return out

class ConvBlock(nn.Module):
    """卷积块: Conv -> 激活 -> MaxPool(尺寸减半), 可带 1x1 卷积残差连接。"""
    def __init__(self, Cin, Cout, K, Pooling=2, padding=None, Batch_norm=False, Res=True, use_tanh=True):
        """参数:
            Cin, Cout: 输入/输出通道数。
            K:         卷积核大小。
            Pooling:   池化倍数。
            Batch_norm: 是否加 BatchNorm; Res: 是否使用残差; use_tanh: 激活函数用 Tanh 还是 LeakyReLU。
        """
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        if not padding:
            P = K//2
        else:
            P = padding
        if Batch_norm:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(Cin),
                nn.Conv2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.MaxPool2d(Pooling),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.MaxPool2d(Pooling),
            )
        # self.downSample = nn.AvgPool2d(2)
        self.shortcut = nn.Sequential(
                     nn.Conv2d(Cin, Cout, kernel_size=1),
                     nn.AvgPool2d(2)
                )
        self.res = Res
        self.cin = Cin
        self.cout = Cout
    
    def forward(self,x):
        """(B, Cin, W, H) -> (B, Cout, W/2, H/2)。"""
        x1 = self.layer(x)
        if self.res:
            x_res = self.shortcut(x)
            x1 = x1 + x_res
        return x1
    
class DeConvBlock(nn.Module):
    """反卷积块(用于图像解码器): 转置卷积 -> 激活 -> 双线性上采样 -> 卷积, 可带残差。"""
    def __init__(self, Cin, Cout, K, upsample, padding=None, Batch_norm=False, Res = True, use_tanh=True):
        """参数含义同 ConvBlock, upsample 为上采样后的目标尺寸。"""
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        if not padding:
            P = K//2
        else:
            P = padding
        if Batch_norm:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(Cin),
                nn.ConvTranspose2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=K,padding=P),
            )
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=K,padding=P),
            )

        self.res = Res
        self.cin = Cin
        self.cout = Cout
        self.short_cut = nn.Sequential(
                nn.ConvTranspose2d(Cin,Cout,kernel_size=1),
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=1),
            )

    def forward(self,x):
        """上采样一次图像特征(要求宽高相等)。"""
        x1 = self.layer(x)
        _, _, W, H = x.shape
        if W != H:
            raise NotImplementedError
        if self.res:
            x_res = self.short_cut(x)
            x1 = x1 + x_res
        return x1

class ImgEncoder(nn.Module):
    """图像编码器: 若干 ConvBlock -> 展平 -> 全连接层 -> 输出均值和(对数)方差两个 embed_size 维向量。"""
    def __init__(self, input_shape, K, embed_size, c_conv_list, size_fc_list,\
                Pooling=2, padding=None, Batch_norm=False, Res=True, use_tanh=True):
        """参数:
            input_shape: (C, W, H)。
            K: 卷积核大小。
            embed_size: 输出向量维度。
            c_conv_list: 每个卷积块的输出通道数, 如 [4, 8]。
            size_fc_list: 全连接层维度, 如 [256]。
        """
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        Cin, w, h = input_shape
        layers = [ConvBlock(Cin, c_conv_list[0], K, Pooling, padding, Batch_norm, Res, use_tanh)]
        for i in range(len(c_conv_list)-1):
            layers.append(ConvBlock(c_conv_list[i], c_conv_list[i+1], K, Pooling, padding, Batch_norm, Res, use_tanh))
        layers.append(nn.Flatten())
        linear_input_shape = int(w*h*c_conv_list[-1]/(4**(len(c_conv_list))))
        layers.extend([nn.Linear(linear_input_shape, size_fc_list[0]), activate_func])
        for i in range(len(size_fc_list)-1):
            layers.append(nn.Linear(size_fc_list[i], size_fc_list[i+1]))
            layers.append(activate_func)
        self.net = nn.Sequential(*layers)
        self.output_mean = nn.Linear(size_fc_list[-1],embed_size)
        self.output_std = nn.Linear(size_fc_list[-1],embed_size)
    
    def forward(self, x):
        """返回 (mean, std) 两个 (B, embed_size) 张量; 在 RL 网络中只使用 mean。"""
        x = self.net(x)
        return self.output_mean(x), self.output_std(x)
    
class ImgDecoder(nn.Module):
    """图像解码器: 把 embed_size 维向量还原为图像, 用于自编码器预训练。"""
    def __init__(self, output_shape, K, embed_size, c_conv_list, size_fc_list,\
                 padding=None, Batch_norm=False, Res=True, use_tanh=True):
        """参数与 ImgEncoder 对应, output_shape 为输出图像形状 (C, W, H)。"""
        super().__init__()
        out_channel, w, h = output_shape
        self.output_shape = output_shape
        self.c_conv_list = c_conv_list
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]

        # The fully connected layers
        fc_list = []
        fc_list.extend([nn.Linear(embed_size, size_fc_list[-1]), activate_func])
        for i in range(len(size_fc_list)-1):
            fc_list.append(nn.Linear(size_fc_list[-(i+1)], size_fc_list[-(i+2)]))
            fc_list.append(activate_func)
        fc_output_size = int( h/(2**len(c_conv_list)) * w/(2**len(c_conv_list)) * c_conv_list[-1] )
        fc_list.extend([nn.Linear(size_fc_list[0], fc_output_size), activate_func])
        self.fc_net = nn.Sequential(*fc_list)

        # The DeConvolution layers
        conv_list = []
        upsample_size = int(h/(2**len(c_conv_list))*2)
        for i in range(len(c_conv_list)-1):
            conv_list.append(DeConvBlock(c_conv_list[-i-1],c_conv_list[-i-2],K,upsample_size, padding, Batch_norm, Res, use_tanh))
            upsample_size *= 2
        conv_list.append(DeConvBlock(c_conv_list[0],out_channel,K,upsample_size, padding, Batch_norm, Res, use_tanh))
        self.conv_net = nn.Sequential(*conv_list)

        self.output = nn.Sigmoid()
    
    def forward(self, z):
        """(B, embed_size) -> (B, C, W, H), 输出经 Sigmoid 在 [0,1]。"""
        x = self.fc_net(z)
        B = x.shape[0]
        # assume the image: w==h
        w = int(self.output_shape[-2]/(2**len(self.c_conv_list)))
        x = x.reshape((B, self.c_conv_list[-1], w, w))
        x = self.conv_net(x)
        x = self.output(x)
        return x
    
class VAE_Conv(nn.Module):
    """卷积变分自编码器(VAE), 用于预训练图像编码器(本仓库未包含其训练脚本)。"""
    def __init__(self, img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=False):
        """构建编码器和解码器。"""
        super(VAE_Conv, self).__init__()
        self.z_size = embed_size
        self.encoder = ImgEncoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        self.decoder = ImgDecoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
    
    def sampling(self, mu, log_var):
        """重参数化采样: z = mu + eps * exp(0.5 * log_var)。"""
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def forward(self, x):
        """编码 -> 采样 -> 解码, 返回 (重建图像, mean, log_var)。"""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,std = self.encoder(x)
        sampled_z = self.sampling(mean,std)
        return self.decoder(sampled_z), mean, std
    
    def eval_forward(self, x):
        """评估用前向: 直接用均值解码(不采样)。"""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,std = self.encoder(x)
        # sampled_z = self.sampling(mean,std)
        return self.decoder(mean), mean, std

    def embed(self,img):
        """无梯度地把图像编码为向量(均值)。"""
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            embed_img,_ = self.encoder(img)
        return embed_img
    
    def save(self, path):
        """保存整个模型对象。"""
        torch.save(self, path)
        print('save model in: %s'%path)

class AE_Conv(nn.Module):
    """卷积自编码器(AE), model/ckpt/autoencoder.pt 即为该类的预训练模型, 其 encoder 被 RL 网络复用。"""
    def __init__(self, img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=False):
        """构建编码器和解码器。"""
        super(AE_Conv, self).__init__()
        self.z_size = embed_size
        self.encoder = ImgEncoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        self.decoder = ImgDecoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        
    def forward(self, x):
        """编码后直接用均值解码, 返回重建图像。"""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,_ = self.encoder(x)
        return self.decoder(mean)

    def embed(self,img):
        """无梯度地把图像编码, 返回 (mean, std)。"""
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            embed_img, embed_std = self.encoder(img)
        return embed_img, embed_std

    def save(self, path):
        """保存整个模型对象。"""
        torch.save(self, path)
        print('save model in: %s'%path)
