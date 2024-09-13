import torch.nn.functional as F
import numpy as np
from mmcv.cnn import Conv2d
import torch
import torch.nn as nn

from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = y.size()
        proj_query = self.query_conv(y).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + y
        return out


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution without padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=0, bias=bias))

def conv3x3_P(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        # 获取两个 Batch Normalization 层的权重的绝对值
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        # 初始化两个与输入张量相同大小的零张量
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        # 根据 Batch Normalization 权重的阈值，交换输入张量的部分值
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        # 通过循环创建指定数量的 Batch Normalization 层，并将它们作为模块的属性
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        # 对输入列表中的每个张量应用相应的 Batch Normalization 层，并返回结果的列表
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, planes, expansion, num_parallel, bn_threshold, stride=1):
        super(Bottleneck, self).__init__()
        self.midplane = planes // expansion
        self.conv1 = conv1x1(planes, self.midplane)
        self.bn1 = BatchNorm2dParallel(self.midplane, num_parallel)
        #self.conv2 = ModuleParallel(DepthwiseSeparableConv(self.midplane, self.midplane, stride=stride))
        self.conv2 = conv3x3_P(self.midplane, self.midplane, stride=stride)
        self.bn2 = BatchNorm2dParallel(self.midplane, num_parallel)
        self.conv21 = conv3x3_P(self.midplane, self.midplane, stride=stride)
        self.bn21 = BatchNorm2dParallel(self.midplane, num_parallel)
        self.conv3 = conv1x1(self.midplane, planes)
        self.bn3 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv21(out)
        out = self.bn21(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)
        return out

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, pooling_r=2):
        super(SCConv, self).__init__()
        self.k1 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(planes))
        self.k2 = nn.Sequential(nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                                nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(planes))
        self.k3 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(planes))
        self.k4 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(planes))
        self.conv1_a = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_a = nn.BatchNorm2d(planes)
        self.conv1_b = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_b = nn.BatchNorm2d(planes)

    def forward(self, x):
        out_a = self.conv1_a(x)
        identity = out_a
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = F.relu(out_a)
        out_b = F.relu(out_b)
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(out_a), identity.size()[2:])))
        out = torch.mul(self.k3(out_a), out)
        out1 = self.k4(out)
        out2 = self.k1(out_b)
        out = torch.cat((out1, out2), 1)
        return out

class External_attention(nn.Module):
    def __init__(self, c):
        super(External_attention, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = c // 4
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))
        self.relu = nn.ReLU()

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        x = self.linear_1(attn)
        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x


class SSM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SSM, self).__init__()
        self.hidden_dim = hidden_dim

        # 前向卷积层，捕获局部光谱特征
        self.forward_conv1d = nn.Conv1d(in_channels=49, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 后向卷积层
        self.backward_conv1d = nn.Conv1d(in_channels=49, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 状态转移矩阵 A 和 B，用于捕获时间序列依赖性
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # 可学习参数 delta，用于调整状态更新
        self.delta_param = nn.Parameter(torch.full((hidden_dim,), 0.1))

    def forward(self, x):
        # x 的形状: [batch_size, seq_length, input_dim]
        Batch, SeqLen, InDim = x.shape
        # print("x1", x.shape)

        # 对输入应用前向卷积，生成特征映射
        x_forward = F.silu(self.forward_conv1d(x.permute(0, 2, 1)))  # [batch_size, hidden_dim, seq_len]
        # print(" x_forward ",  x_forward .shape)

        # 对输入应用后向卷积，生成特征映射
        x_backward = F.silu(self.backward_conv1d(torch.flip(x.permute(0, 2, 1), dims=[2])))
        # print("x_backward ", x_backward.shape)
        # 扩展 delta 参数，用于与状态转移矩阵进行运算
        delta_expanded = self.delta_param.unsqueeze(0).unsqueeze(2)

        # 计算前向和后向的状态更新输出
        forward_ssm_output = torch.tanh(x_forward + torch.matmul(self.A, delta_expanded))
        backward_ssm_output = torch.tanh(x_backward + torch.matmul(self.B, delta_expanded))

        # 对特征映射进行池化，得到全局特征
        forward_reduced = forward_ssm_output.mean(dim=2)
        backward_reduced = backward_ssm_output.mean(dim=2)

        # 合并前向和后向的输出，生成最终特征
        output = forward_reduced + backward_reduced
        # print(" output ",  output.shape)

        return output


class GroupWiseSpectralEmbeddingSSM(nn.Module):
    def __init__(self, num_groups, group_size, in_channels, hidden_dim):
        super(GroupWiseSpectralEmbeddingSSM, self).__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        self.ssm_layers = nn.ModuleList([
            SSM(input_dim=group_size, hidden_dim=hidden_dim)
            for _ in range(num_groups)
        ])

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        actual_groups = min(self.num_groups, channels)
        group_size = channels // actual_groups
        x = x[:, :actual_groups * group_size, :, :]  # 确保输入可以被整除
        x = x.view(batch_size, actual_groups, group_size, height * width)
        outputs = []
        for i in range(actual_groups):
            group_features = x[:, i, :, :]  # 提取第 i 组的特征
            out = self.ssm_layers[i](group_features)  # 通过 SSM 进行处理
            # print("out1 ",out.shape)
            #out = out.view(batch_size, self.hidden_dim, height, width)  # 恢复到 [batch_size, hidden_dim, height, width]
            outputs.append(out)
        return torch.cat(outputs, dim=1)  # 在通道维度上拼接输出




# class HSIVimBlock(nn.Module):
#     def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_groups, group_size):
#         super(HSIVimBlock, self).__init__()
#         self.spatial_dim = spatial_dim
#         self.num_bands = num_bands
#         self.hidden_dim = hidden_dim
#
#         # 局部特征提取模块
#         self.gse = GroupWiseSpectralEmbeddingSSM(num_groups, group_size, in_channels=1, hidden_dim=hidden_dim)
#
#         # # 全局特征提取模块
#         # self.norm = nn.LayerNorm(num_bands * spatial_dim * spatial_dim)
#         # self.linear_x = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)
#         # self.linear_z = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)
#         #
#         # self.forward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
#         # self.backward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
#         #
#         # self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
#         # self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
#         # self.delta_param = nn.Parameter(torch.full((hidden_dim,), delta_param_init))
#         #
#         # self.linear_forward = nn.Linear(hidden_dim, output_dim)
#         # self.linear_backward = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         Batch, H, W, Bands = x.shape  # 获取输入的形状 [Batch, Height, Width, Bands]
#
#         # 应用局部特征提取
#         local_features = self.gse(x)  # 提取局部特征
#
#         # 将局部特征展平成一维向量
#         local_features_flattened = local_features.reshape(Batch, -1)  # 展平为 [Batch, hidden_dim*H*W]
#
#         # # # 进行 LayerNorm 处理
#         # global_features = x.reshape(Batch, -1)  # 展平为 [Batch, Bands*H*W]
#         # expected_shape = self.num_bands * self.spatial_dim * self.spatial_dim
#         # if global_features.shape[1] != expected_shape:
#         #     raise ValueError(f"Expected input shape [{Batch}, {expected_shape}], but got {global_features.shape}")
#         #
#         # global_features = self.norm(global_features)
#         #
#         # # 投影到隐藏维度
#         # x_proj = self.linear_x(global_features)
#         # z_proj = self.linear_z(global_features)
#         #
#         # # 调整形状以适应 Conv1d
#         # x_proj = x_proj.view(Batch, self.hidden_dim, -1)
#         # z_proj = z_proj.view(Batch, self.hidden_dim, -1)
#         # # 翻转 z_proj 以用于后向处理
#         # z_proj_reversed = torch.flip(z_proj, dims=[-1])
#         #
#         # # 双向 Conv1d 处理
#         # x_forward = F.silu(self.forward_conv1d(x_proj))
#         # x_backward = F.silu(self.backward_conv1d(z_proj_reversed))
#         #
#         # # 正确应用 delta 参数
#         # delta_expanded = self.delta_param.unsqueeze(0).unsqueeze(2)
#         #
#         # # 进行 SSM 处理，分别使用原始和翻转的输入用于前向和后向路径
#         # forward_ssm_output = torch.tanh(self.forward_conv1d(x_forward) + self.A * delta_expanded)
#         # backward_ssm_output = torch.tanh(self.backward_conv1d(x_backward) + self.B * delta_expanded)
#         #
#         # # 将前向和后向输出组合为一个表示
#         # forward_reduced = forward_ssm_output.mean(dim=2)
#         # backward_reduced = backward_ssm_output.mean(dim=2)
#         #
#         # # 将减少的前向和后向路径组合
#         # y_forward = self.linear_forward(forward_reduced)
#         # y_backward = self.linear_backward(backward_reduced)
#         #
#         # # 元素级求和前向和后向输出
#         # global_features_combined = y_forward + y_backward
#
#         # 将局部特征展平并投影到全局特征的维度
#         local_features_flattened_proj = nn.Linear(local_features_flattened.shape[1], self.hidden_dim).to(x.device)(
#             local_features_flattened)
#
#         # 将局部特征直接与全局特征相加
#         # output = global_features_combined + local_features_flattened_proj
#         output = local_features_flattened_proj
#
#         return output


class GroupWiseSpectralEmbedding(nn.Module):
    def __init__(self, num_groups, group_size, in_channels, out_channels):
        super(GroupWiseSpectralEmbedding, self).__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, group_size), stride=1, padding=(0, group_size // 2))
            for _ in range(num_groups)
        ])

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # 计算实际的分组数量，避免出现组数为零的情况
        actual_groups = min(self.num_groups, channels)
        group_size = channels // actual_groups
        x = x[:, :actual_groups * group_size, :, :]  # 确保输入可以被整除
        x = x.view(batch_size, actual_groups, group_size, height, width)
        outputs = []
        for i in range(actual_groups):
            out = self.conv_layers[i](x[:, i, :, :, :])
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class HSIVimBlock(nn.Module):
    def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_groups, group_size):
        super(HSIVimBlock, self).__init__()
        self.spatial_dim = spatial_dim
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim

        # 局部特征提取模块
        self.gse = GroupWiseSpectralEmbedding(num_groups, group_size, in_channels=1, out_channels=1)

        # 全局特征提取模块
        self.norm = nn.LayerNorm(num_bands * spatial_dim * spatial_dim)
        self.linear_x = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)
        self.linear_z = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)

        self.forward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.backward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.delta_param = nn.Parameter(torch.full((hidden_dim,), delta_param_init))

        self.linear_forward = nn.Linear(hidden_dim, output_dim)
        self.linear_backward = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        Batch, H, W, Bands = x.shape  # Correct shape extraction assuming [Batch, Height, Width, Bands]

        # Apply GroupWise Spectral Embedding
        x = x.permute(0, 3, 1, 2)  # Reorder to [Batch, Bands, Height, Width]
        x = self.gse(x)
        x = x.permute(0, 2, 3, 1)  # Reorder back to [Batch, Height, Width, Bands]

        # Correctly reshape for LayerNorm to flatten all spatial and spectral information
        x = x.reshape(Batch, -1)  # New shape: [Batch, Bands*H*W]

        # 调整形状确保符合LayerNorm的输入要求
        expected_shape = self.num_bands * self.spatial_dim * self.spatial_dim
        if x.shape[1] != expected_shape:
            raise ValueError(f"Expected input shape [{Batch}, {expected_shape}], but got {x.shape}")

        x = self.norm(x)

        # Projection to hidden dimensions
        x_proj = self.linear_x(x)
        z_proj = self.linear_z(x)

        # Ensure correct reshaping for Conv1d compatibility
        x_proj = x_proj.view(Batch, self.hidden_dim, -1)
        z_proj = z_proj.view(Batch, self.hidden_dim, -1)

        # Reverse z_proj for the backward path
        z_proj_reversed = torch.flip(z_proj, dims=[-1])

        # Bidirectional Conv1d processing using reversed input for the backward path
        x_forward = F.silu(self.forward_conv1d(x_proj))
        x_backward = F.silu(self.backward_conv1d(z_proj_reversed))

        # Apply delta parameter correctly
        delta_expanded = self.delta_param.unsqueeze(0).unsqueeze(2)  # Correct shape for broadcasting

        # SSM processing with delta applied, using the original and reversed inputs for forward and backward paths respectively
        forward_ssm_output = torch.tanh(self.forward_conv1d(x_proj) + self.A * delta_expanded)
        backward_ssm_output = torch.tanh(self.backward_conv1d(z_proj_reversed) + self.B * delta_expanded)

        # Combine forward and backward outputs into a single representation
        forward_reduced = forward_ssm_output.mean(dim=2)
        backward_reduced = backward_ssm_output.mean(dim=2)

        # Combine the reduced forward and backward paths
        y_forward = self.linear_forward(forward_reduced)
        y_backward = self.linear_backward(backward_reduced)

        # Element-wise sum of forward and backward outputs
        y_combined = y_forward + y_backward

        return y_combined







class HSIClassificationMambaModel(nn.Module):
    def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_classes, num_groups, group_size):
        super(HSIClassificationMambaModel, self).__init__()
        self.vim_block = HSIVimBlock(spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_groups, group_size)
        self.output_dim = output_dim  # Save output_dim as an attribute of the class
        # self.spatial_processing = SpatialFeatureProcessing(input_channels=output_dim)
        # self.classifier = HSIClassifier(in_features=512, num_classes=num_classes)

    def forward(self, x):
        x = self.vim_block(x)
        x = x.view(-1, self.output_dim, 1, 1)  # Reshape to include spatial dimensions if needed
        # x = self.spatial_processing(x)
        # Flatten the output from vim_block
        x = torch.flatten(x, start_dim=1)
        # x = self.classifier(x)
        return x


##整体网络
class DynamicFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(DynamicFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim1, output_dim)
        self.fc2 = nn.Linear(input_dim2, output_dim)
        self.gamma = nn.Parameter(torch.rand(1))  # learnable parameter for dynamic fusion

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        out = self.gamma * x1 + (1 - self.gamma) * x2
        return out

class FinalModel(nn.Module):
    def __init__(self, hsi_channels, msi_channels, hidden_size, Bottleneck, num_parallel, num_reslayer, num_classes,
                bn_threshold, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_classes2, num_groups, group_size):
        super(FinalModel, self).__init__()
        self.hsi_model = HSIClassificationMambaModel(spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init,
                                                     num_classes2, num_groups, group_size)
        self.hsi_msi_model = HSIMSINet(hsi_channels, msi_channels, hidden_size, Bottleneck, num_parallel, num_reslayer,
                                       num_classes, bn_threshold)
        self.dynamic_fusion = DynamicFusion(output_dim, hidden_size * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_train2, X_train, X_train_2):
        final_output= self.hsi_model(X_train2)
        final_output, _ = self.hsi_msi_model(X_train, X_train_2)
        # combined_features = self.dynamic_fusion(hsi_features, hsi_msi_features)
        # final_output = self.fc2(F.relu(combined_features))
        return final_output



class HSIMSIClassifier(nn.Module):
    def __init__(self,  hidden_size, num_classes):
        super(HSIMSIClassifier, self).__init__()
        # 全局平均池化层，将特征图降维为 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 第一个全连接层，输入维度为 hidden_size*2，输出维度为 hidden_size
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        # 第二个全连接层，输入维度为 hidden_size，输出维度为 num_classes
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout()
    def forward(self, x):
        # 全局平均池化
        x = self.avg_pool(x)
        # 将特征图展平为一维张量
        x = x.view(x.size(0), -1)
        # 第一个全连接层，激活函数使用 ReLU
        out = self.fc2(F.relu(self.fc1(x)))
        return out


class HSIClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(HSIClassifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x):

        x = self.fc_layers(x)

        # 如果使用CrossEntropyLoss，可以去掉Softmax
        return x




# class HSIClassificationMambaModel(nn.Module):
#     def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_classes, num_groups, group_size):
#         super(HSIClassificationMambaModel, self).__init__()
#         self.vim_block = HSIVimBlock(spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_groups, group_size)
#         self.output_dim = output_dim  # Save output_dim as an attribute of the class
#
#     def forward(self, x):
#         x = self.vim_block(x)
#         x = x.view(-1, self.output_dim, 1, 1)  # Reshape to include spatial dimensions if needed
#         x = torch.flatten(x, start_dim=1)
#         return x

class HSIMSINet(nn.Module):
    def __init__(self, hsi_channels, msi_channels, hidden_size, block, num_parallel, num_reslayer=2, num_classes=16, bn_threshold=2e-2):
        super(HSIMSINet, self).__init__()
        self.planes = hidden_size
        self.num_parallel = num_parallel
        self.expansion = 2
        self.conv_00 = nn.Sequential(nn.Conv2d(hsi_channels, hidden_size, 1, bias=False), nn.BatchNorm2d(hidden_size))
        self.conv_11 = nn.Sequential(nn.Conv2d(msi_channels, hidden_size, 1, bias=False), nn.BatchNorm2d(hidden_size))
        self.conv1 = ModuleParallel(DepthwiseSeparableConv(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0, bias=False))
        #self.conv1 = ModuleParallel(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0, bias=False))
        self.bn1 = BatchNorm2dParallel(hidden_size, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.layer = self._make_layer(block, hidden_size, num_reslayer, bn_threshold)
        self.Attention = External_attention(hidden_size * 2)
        self.SCConv = SCConv(hidden_size * 2, hidden_size)
        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)
        self.PAMmodel = PAM_Module(in_dim=256)
        # Global Average Pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = HSIMSIClassifier(hidden_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        layers = []
        layers.append(block(planes, self.expansion, self.num_parallel, bn_threshold, stride))
        for i in range(1, num_blocks):
            layers.append(block(planes, planes, self.num_parallel, bn_threshold))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        y = F.relu(self.conv_11(y)).unsqueeze(0)
        x = F.relu(self.conv_00(x)).unsqueeze(0)
        x = torch.cat((x, y), 0)
        x = self.relu(self.bn1(self.conv1(x)))
        out = self.layer(x)
        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * out[l].detach()
        out.append(ens)
        x = torch.cat((out[0], out[1]), dim=1)
        x = self.Attention(x)
        # print("Shape of x4 after concatenation:", x.shape)
        x = self.PAMmodel(x)

        x= self.SCConv(x)
        x = self.avg_pool(x)
        out= x.view(x.size(0), -1)
        out = self.classifier(x)
        return out, alpha_soft
