import os
import torch
import torch.optim
from torch import nn
from models.ClassifierNet import  FinalModel, Bottleneck
from data_loader import build_datasets
import args_parser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import time
args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(args)


def calc_loss(outputs, labels):
    # 使用交叉熵损失作为损失函数
    criterion = nn.CrossEntropyLoss()
    # 计算交叉熵损失
    loss = criterion(outputs, labels)
    return loss

def L1_penalty(var):
    # 计算张量 var 的 L1 范数
    return torch.abs(var).sum()


def train(model, device, train_loader, HSI_train_loader, optimizer, epoch, slim_params, bn_threshold):
    model.train()
    total_loss = 0
    num_batches = 0

    # 合并训练 train_loader 和 HSI_train_loader 的逻辑
    for i, (inputs_1, inputs_2, labels) in enumerate(train_loader):
        inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        final_output = model(X_train2=inputs_1,X_train=inputs_1,X_train_2=inputs_2)#X_train, X_train_2,
        # final_output, _  = model(inputs_1, inputs_2)
        loss = calc_loss(final_output, labels)
        L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
        loss += 0.05 * L1_norm
    #反向传播优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1


    slim_params_list = []
    for slim_param in slim_params:
        slim_params_list.extend(slim_param.cpu().data.numpy())
    slim_params_list = np.array(sorted(slim_params_list))

    if len(slim_params_list) > 0:
        print('Epoch %d, 3%% smallest slim_params: %.4f' % (epoch, slim_params_list[len(slim_params_list) // 33]), flush=True, end=" ")
    else:
        print('Epoch %d, slim_params_list is empty.' % epoch, flush=True, end=" ")

    print('  [loss avg: %.4f]' % (total_loss / num_batches))


def val(model, device, val_loader, HSI_val_loader):
    model.eval()
    count = 0

    # 遍历 val_loader 数据集
    for inputs_1, inputs_2, labels in val_loader:
        inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
        labels = labels.to(device)

        # 禁用梯度计算
        with torch.no_grad():
            final_output= model( inputs_1, inputs_1,inputs_2)
            #final_output = model(X_train2=inputs_1, X_train=inputs_1, X_train_2=inputs_2)
            outputs = np.argmax(final_output.detach().cpu().numpy(), axis=1)

        if count == 0:
            y_pred_val = outputs
            val_labels = labels.cpu().numpy()
            count = 1
        else:
            y_pred_val = np.concatenate((y_pred_val, outputs))
            val_labels = np.concatenate((val_labels, labels.cpu().numpy()))

    # # 遍历 HSI_val_loader 数据集
    # for inputs_1, labels in HSI_val_loader:
    #     inputs_1 = inputs_1.to(device)
    #     labels = labels.to(device)
    #
    #     with torch.no_grad():
    #         zero_input = torch.zeros((inputs_1.size(0), 3, inputs_1.size(2), inputs_1.size(3)), device=device)
    #         final_output = model(X_train2=inputs_1, X_train=inputs_1, X_train_2=zero_input)
    #         outputs = np.argmax(final_output.detach().cpu().numpy(), axis=1)
    #
    #     if count == 0:
    #         y_pred_val = outputs
    #         val_labels = labels.cpu().numpy()
    #         count = 1
    #     else:
    #         y_pred_val = np.concatenate((y_pred_val, outputs))
    #         val_labels = np.concatenate((val_labels, labels.cpu().numpy()))

    # 计算分类准确率
    a = 0
    for c in range(len(y_pred_val)):
        if val_labels[c] == y_pred_val[c]:
            a += 1
    acc = a / len(y_pred_val) * 100

    # 打印准确率
    print(' [The verification accuracy is: %.2f]' % acc)

    return acc


def main():
    train_loader, val_loader, HSI_train_loader, HSI_val_loader = build_datasets(args.root, args.dataset,args.patch_size, args.batch_size,args.val_ratio)
    if args.dataset == 'Houston2018':
        args.hsi_bands = 144
        args.msi_bands = 8
        args.num_class = 16
    elif args.dataset == 'HAIDONG':
        args.hsi_bands = 32
        args.msi_bands = 3
        args.num_class = 12
    elif args.dataset == 'HHK':
        args.hsi_bands = 166
        args.msi_bands = 3
        args.num_class = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 构建数据加载器




    # 初始化模型
    model =FinalModel(args.hsi_bands, args.msi_bands, args.hidden_size, Bottleneck, args.num_parallel, args.num_reslayer,
                args.num_class, args.bn_threshold,spatial_dim=7,num_bands=144, hidden_dim=256,output_dim=256,delta_param_init=0.01, num_classes2=16,num_groups =130,
    group_size =11).eval().to(device)

    params_to_update = list(model.parameters())
    betas = (0.9, 0.999)  # betas of Adam
    # optimizer = torch.optim.Adam(params_to_update, lr=args.lr, betas=betas, eps=1e-8, weight_decay=0.0009)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.94, weight_decay=0.07)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    Net_params, slim_params = [], []
    # 提取需要进行L1正则化的参数
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith('weight') and 'bn2' in name:
            if len(slim_params) % 2 == 0:
                slim_params.append(param[:len(param) // 2])
            else:
                slim_params.append(param[len(param) // 2:])

    best_acc = 0
    for epoch in range(args.epochs):
        train(model, device, train_loader, HSI_train_loader, optimizer, epoch, slim_params, args.bn_threshold)

        # 每两个epoch进行一次测试，并保存最佳模型
        if (epoch + 1) % 2 == 0:
            acc = val(model, device, val_loader,HSI_val_loader)
            if acc >= best_acc:
                best_acc = acc
                print("保存模型")
                torch.save(model.state_dict(), r"E:\code2024\spa and spe\HSI-MSI（data public）\models\model10e-65.pth")





        # 清理缓存
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
