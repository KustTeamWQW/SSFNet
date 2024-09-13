# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from mmcv.cnn import Conv2d
# import torch
# import torch.nn as nn
#
# from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
#     NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
#
# # class PAM_Module(nn.Module):
# #     """ Position attention module"""
# #     #Ref from SAGAN
# #     def __init__(self, in_dim):
# #         super(PAM_Module, self).__init__()
# #         self.chanel_in = in_dim
# #
# #         self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
# #         self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
# #         self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
# #         self.gamma = Parameter(torch.zeros(1))
# #
# #         self.softmax = Softmax(dim=-1)
# #     def forward(self, y):
# #         """
# #             inputs :
# #                 x : input feature maps( B X C X H X W)
# #             returns :
# #                 out : attention value + input feature
# #                 attention: B X (HxW) X (HxW)
# #         """
# #         m_batchsize, C, height, width = y.size()
# #         proj_query = self.query_conv(y).view(m_batchsize, -1, width*height).permute(0, 2, 1)
# #         proj_key = self.key_conv(y).view(m_batchsize, -1, width*height)
# #         energy = torch.bmm(proj_query, proj_key)
# #         attention = self.softmax(energy)
# #         proj_value = self.value_conv(y).view(m_batchsize, -1, width*height)
# #
# #         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
# #         out = out.view(m_batchsize, C, height, width)
# #
# #         out = self.gamma*out + y
# #         return out
#
#
# # class CAM_Module(nn.Module):
# #     """ Channel attention module"""
# #     def __init__(self, in_dim):
# #         super(CAM_Module, self).__init__()
# #         self.chanel_in = in_dim
# #
# #
# #         self.gamma = Parameter(torch.zeros(1))
# #         self.softmax  = Softmax(dim=-1)
# #     def forward(self,x):
# #         """
# #             inputs :
# #                 x : input feature maps( B X C X H X W)
# #             returns :
# #                 out : attention value + input feature
# #                 attention: B X C X C
# #         """
# #         m_batchsize, C, height, width = x.size()
# #         proj_query = x.view(m_batchsize, C, -1)
# #         proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
# #         energy = torch.bmm(proj_query, proj_key)
# #         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
# #         attention = self.softmax(energy_new)
# #         proj_value = x.view(m_batchsize, C, -1)
# #
# #         out = torch.bmm(attention, proj_value)
# #         out = out.view(m_batchsize, C, height, width)
# #
# #         out = self.gamma*out + x
# #         return out
# #
# #
#
# # def conv3x3(in_planes, out_planes, stride=1, bias=False):
# #     "3x3 convolution without padding"
# #     return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
# #                                     stride=stride, padding=0, bias=bias))
# class SeparableConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super(SeparableConv2D, self).__init__()
#
#         # 定义深度卷积操作
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
#
#         # 定义逐点卷积操作
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
#
#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#
#         return out
# def sep2d_residual_block(input_layer, filters, kernel_size):
#     first_layer = SeparableConv2D(filters=filters, kernel_size=kernel_size,
#                                   activation='relu', padding='same')(input_layer)
#     x = SeparableConv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(first_layer)
#     x = SeparableConv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
#     x = Add()([x, first_layer])
#
#     return x
# def conv3x3_p(in_planes, out_planes, stride=1, bias=False):
#     "3x3 convolution with padding"
#     return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=bias))
#
# def conv1x1(in_planes, out_planes, stride=1, bias=False):
#     "1x1 convolution"
#     return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
#                                     stride=stride, padding=0, bias=bias))
#
#
# class Exchange(nn.Module):
#     def __init__(self):
#         super(Exchange, self).__init__()
#
#     def forward(self, x, bn, bn_threshold):
#         # 获取两个 Batch Normalization 层的权重的绝对值
#         bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
#         # 初始化两个与输入张量相同大小的零张量
#         x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
#         # 根据 Batch Normalization 权重的阈值，交换输入张量的部分值
#         x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
#         x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
#         x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
#         x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
#         return [x1, x2]
#
# class ModuleParallel(nn.Module):
#     def __init__(self, module):
#         super(ModuleParallel, self).__init__()
#         self.module = module
#
#     def forward(self, x_parallel):
#         return [self.module(x) for x in x_parallel]
#
# class BatchNorm2dParallel(nn.Module):
#     def __init__(self, num_features, num_parallel):
#         super(BatchNorm2dParallel, self).__init__()
#         # 通过循环创建指定数量的 Batch Normalization 层，并将它们作为模块的属性
#         for i in range(num_parallel):
#             setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))
#     def forward(self, x_parallel):
#         # 对输入列表中的每个张量应用相应的 Batch Normalization 层，并返回结果的列表
#         return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
#
# # class Bottleneck(nn.Module):
# #     def __init__(self):
# #         super(Bottleneck, self).__init__()
# #
# #         self.conv1 = SeparableConv2D(3, 64, 3, padding=1)
# #         self.conv2 = SeparableConv2D(64, 64, 3, padding=1)
# #         self.conv3 = SeparableConv2D(64, 128, 3, padding=1)
# #         self.fc = nn.Linear(128 * 32 * 32, 10)
# #
# #     def forward(self, x):
# #         out = self.conv1(x)
# #         out = nn.ReLU()(out)
# #         out = nn.MaxPool2d(2)(out)
# #
# #         out = self.conv2(out)
# #         out = nn.ReLU()(out)
# #         out = nn.MaxPool2d(2)(out)
# #
# #         out = self.conv3(out)
# #         out = nn.ReLU()(out)
# #         out = nn.MaxPool2d(2)(out)
# #
# #         out = out.view(out.size(0), -1)
# #
# #         out = self.fc(out)
# #
# #         return out
#
# class Bottleneck(nn.Module):
#
#     def __init__(self, planes,expansion, num_parallel, bn_threshold, stride=1):
#         super(Bottleneck, self).__init__()
#         # 计算中间通道数
#         self.midplane = planes//expansion
#         # 第一个卷积层：1x1 卷积(压缩维度即减少通道数)
#         self.conv1 = conv1x1(planes, self.midplane)
#         self.bn1 = BatchNorm2dParallel(self.midplane, num_parallel)
#         # 第二个卷积层：3x3 并行卷积(捕获特征之间的空间关系并进行非线性映射)
#         self.conv2 = conv3x3_p(self.midplane, self.midplane, stride=stride)
#         self.bn2 = BatchNorm2dParallel(self.midplane, num_parallel)
#         # 第三个卷积层：1x1 卷积(恢复维度)
#         self.conv3 = conv1x1(self.midplane, planes)
#         self.bn3 = BatchNorm2dParallel(planes, num_parallel)
#         # ReLU 激活函数
#         self.relu = ModuleParallel(nn.ReLU(inplace=True))
#         # 记录并行数和 Batch Normalization 层的阈值
#         self.num_parallel = num_parallel
#         self.exchange = Exchange()
#         self.bn_threshold = bn_threshold
#         # 获取第二个 Batch Normalization 层的列表
#         self.bn2_list = []
#         for module in self.bn2.modules():
#             if isinstance(module, nn.BatchNorm2d):
#                 self.bn2_list.append(module)
#
#     def forward(self, x):
#         residual = x
#         out = x
#         # 第一个卷积层：1x1 卷积
#         out = self.conv1(out)
#         out = self.bn1(out)
#         out = self.relu(out)
#         # 第二个卷积层：3x3 并行卷积
#         out = self.conv2(out)
#         out = self.bn2(out)
#         # if len(x) > 0:
#         # Batch Normalization 交换
#         out = self.exchange(out, self.bn2_list, self.bn_threshold)
#         out = self.relu(out)
#         # 第三个卷积层：1x1 卷积
#         out = self.conv3(out)
#         out = self.bn3(out)
#         # 残差连接
#         out = [out[l] + residual[l] for l in range(self.num_parallel)]
#         out = self.relu(out)
#
#         return out
#
# class Dropout(nn.Module):
#     def __init__(self):
#         super(Dropout, self).__init__()
#         # Dropout 模块可以用于在深度神经网络中引入随机性，从而有助于防止过拟合。
#     def forward(self, x):
#         # 使用 F.dropout 实现 dropout 操作，p=0.2 表示丢弃概率为 0.2
#         out = F.dropout(x, p=0.2, training=self.training)
#         return out
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, size, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.eps = eps
#         # 创建可学习的参数 a_2 和 b_2，分别用于缩放和平移
#         self.a_2 = nn.Parameter(torch.ones(size))
#         self.b_2 = nn.Parameter(torch.zeros(size))
#     def forward(self, x):
#         # 计算输入张量 x 在最后一个维度上的均值和标准差
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         # 使用 Layer Normalization 公式进行归一化
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
#
#
#
# class SCConv(nn.Module):
#     def __init__(self, inplanes, planes, pooling_r =2):
#         super(SCConv, self).__init__()
#         # 定义四个卷积块 k1、k2、k3、k4
#         self.k1 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,  padding =1, bias=False),
#                     nn.BatchNorm2d(planes),)
#         self.k2 = nn.Sequential(nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
#                     nn.Conv2d(planes, planes, kernel_size=3, padding = 1, bias=False),
#                     nn.BatchNorm2d(planes), )
#         self.k3 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding = 1, bias=False),
#                     nn.BatchNorm2d(planes),)
#         self.k4 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,  padding = 1, bias=False),
#                     nn.BatchNorm2d(planes),)
#         # 定义两个 1x1 卷积层和 Batch Normalization 层
#         self.conv1_a = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1_a = nn.BatchNorm2d(planes)
#         self.conv1_b = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1_b = nn.BatchNorm2d(planes)
#
#     def forward(self, x):
#         # 对输入进行两个分支的卷积操作
#         out_a = self.conv1_a(x)
#         identity = out_a
#         out_a = self.bn1_a(out_a)
#         out_b = self.conv1_b(x)
#         out_b = self.bn1_b(out_b)
#         out_a = F.relu(out_a)
#         out_b = F.relu(out_b)
#         # 进行 Spatial-Channel Concatenation 操作
#         out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(out_a), identity.size()[2:])))
#         out = torch.mul(self.k3(out_a), out)
#         out1 = self.k4(out)
#         out2 = self.k1(out_b)
#         out = torch.cat((out1,out2),1)
#         return out
#
# class External_attention(nn.Module):
#
#     def __init__(self, c):
#         super(External_attention, self).__init__()
#         # 第一个卷积层：1x1 卷积
#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.k = c//4 # 计算 attention 时的降维因子
#         # 第一个线性层：1x1 卷积，用于计算 attention
#         self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
#         # 第二个线性层：1x1 卷积，用于调整权重
#         self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
#         # self.linear_1.weight = self.linear_0.weight.permute(1, 0, 2)
#         # 第二个卷积层：1x1 卷积，带 Batch Normalization
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             nn.BatchNorm2d(c))
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         idn = x# 保存输入作为残差连接的标识符
#         x = self.conv1(x) # 1x1 卷积
#         b, c, h, w = x.size()
#         x = x.view(b, c, h*w) # 展平成二维张量，用于计算 attention
#         attn = self.linear_0(x)# 计算 attention 分数
#         attn = F.softmax(attn, dim=-1)# 使用 softmax 计算注意力权重
#         attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))# 归一化
#         x = self.linear_1(attn) # 通过调整权重
#         x = x.view(b, c, h, w)  # 恢复形状
#         x = self.conv2(x) # 1x1 卷积
#         x = x + idn # 残差连接
#         x = self.relu(x)# ReLU 激活
#         return x
#
#
#
# class Classifier(nn.Module):
#     def __init__(self,  hidden_size, num_classes):
#         super(Classifier, self).__init__()
#         # 全局平均池化层，将特征图降维为 1x1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # 第一个全连接层，输入维度为 hidden_size*2，输出维度为 hidden_size
#         self.fc1 = nn.Linear(hidden_size*2, hidden_size)
#         # 第二个全连接层，输入维度为 hidden_size，输出维度为 num_classes
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         # Dropout 层，用于防止过拟合
#         self.dropout = nn.Dropout()
#
#     def forward(self, x):
#         # 全局平均池化
#         x = self.avg_pool(x)
#         # 将特征图展平为一维张量
#         x = x.view(x.size(0), -1)
#         # 第一个全连接层，激活函数使用 ReLU
#         out = self.fc2(F.relu(self.fc1(x)))
#         return out
#
#
# class Net(nn.Module):
#     def __init__(self, hsi_channels, sar_channels, hidden_size, block, num_parallel, num_reslayer=2, num_classes=7, bn_threshold=2e-2):
#         self.planes = hidden_size
#         self.num_parallel = num_parallel
#         self.expansion = 2
#
#         super(Net, self).__init__()
#
#         # HSI (高光谱影像) 路径
#         self.conv_00 = nn.Sequential(nn.Conv2d(hsi_channels, hidden_size, 1, bias=False),#HSI(CAM)
#             nn.BatchNorm2d(hidden_size))
#         # SAR (合成孔径雷达) 路径
#         self.conv_11 = nn.Sequential(nn.Conv2d(sar_channels, hidden_size, 1, bias=False),#MSI(PAM)
#             nn.BatchNorm2d(hidden_size))
#         # # PAM (像素注意力模块) 和 CAM (通道注意力模块)
#         # self.PAMmodel=PAM_Module(in_dim=128)
#         # self.CAMmodul=CAM_Module(in_dim=128)
#         # 共享的卷积和批量归一化层
#         self.conv1 = ModuleParallel(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0, bias=False))
#         self.bn1 = BatchNorm2dParallel(hidden_size, num_parallel)
#         self.relu = ModuleParallel(nn.ReLU(inplace=True))
#         # Residual layers
#         self.layer = self._make_layer(block, hidden_size, num_reslayer, bn_threshold)
#         # 分类器
#         self.classifier = Classifier(hidden_size , num_classes)
#         # 注意力机制和空间通道连接
#         self.Attention = External_attention(hidden_size*2)
#         self.SCConv = SCConv(hidden_size*2, hidden_size)
#         # 超参数用于控制注意力权重
#         self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
#         self.register_parameter('alpha', self.alpha)
#
#     def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
#         layers = []
#         layers.append(block(planes, self.expansion, self.num_parallel, bn_threshold, stride))
#         for i in range(1, num_blocks):
#             layers.append(block(planes, planes, self.num_parallel, bn_threshold))
#         return nn.Sequential(*layers)
#
#     def forward(self, x, y):
#         #y = F.relu(self.conv_11(y)).unsqueeze(0)
#         #y=self.PAMmodel(y)
#
#         # x = F.relu(self.conv_00(x)).unsqueeze(0)
#         # x = F.relu(self.conv_00(x)).unsqueeze(0)
#         # x=self.CAMmodul(x)
#         # y = F.relu(self.conv_11(y)).unsqueeze(0)
#         # y=self.PAMmodel(y)
#         # x = x.squeeze(0)
#         #x=self.CAMmodul(x)
#         #y = F.relu(self.conv_11(y))
#         # y = y.squeeze(0)
#         #y=self.PAMmodel(y)
#         # 对 HSI 和 SAR 进行初始处理
#         x = F.relu(self.conv_00(x))
# #        x = self.CAMmodul(x)
#         x = x.unsqueeze(0)
#
#
#         y = F.relu(self.conv_11(y))
# #        y = self.PAMmodel(y)
#         y = y.unsqueeze(0)
#
#         # 拼接 HSI 和 SAR
#         x = torch.cat((x, y), 0)
#         # 共享的卷积和批量归一化层
#         x = self.relu(self.bn1(self.conv1(x)))
#         # Residual layers
#         out = self.layer(x)
#
#         # 带有注意力权重的输出集成
#         ens = 0
#         alpha_soft = F.softmax(self.alpha)
#
#         for l in range(self.num_parallel):
#             ens += alpha_soft[l] * out[l].detach()
#         out.append(ens)
#         # 空间通道连接和注意力机制
#         x = torch.cat((out[0], out[1]),dim =1)#dim=1 在这里表示沿着通道维度进行拼接
#         x = self.SCConv(self.Attention(x))
#         # 最终预测
#         out = self.classifier(x)
#
#         return out, alpha_soft

