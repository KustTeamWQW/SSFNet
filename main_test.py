import os
import torch
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from models.ClassifierNet import FinalModel, Bottleneck
import args_parser
import spectral as spy
import scipy.io
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from tabulate import tabulate

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(args)

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def addZeroPadding(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX

def main():
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

    # data_hsi = scio.loadmat("E:/code2024/HSI-MSI2/data/HAIDONG/HSI.mat")["HSI"]
    # data_msi = scio.loadmat("E:/code2024/HSI-MSI2/data/HAIDONG/MSI.mat")["MSI"]
    # data_gt = scio.loadmat("E:/code2024/HSI-MSI2/data/HAIDONG/MSI_label.mat")["MSI_label"]
    # data_hsi = np.transpose(data_hsi, (1, 2, 0))
    data_hsi = scio.loadmat("E:\code2024\spa and spe\HSI-MSI（data public）\data\Houston2018\data_HS_LR.mat")["data_HS_LR"]
    data_msi = scio.loadmat("E:\code2024\spa and spe\HSI-MSI（data public）\data\Houston2018\data_MS_HR.mat")["data_MS_HR"]
    data_gt = scio.loadmat(r"E:\code2024\spa and spe\HSI-MSI（data public）\data\Houston2018\Sample.mat")["label"]

    height, width, c = data_hsi.shape
    data_hsi = minmax_normalize(data_hsi)
    data_msi = minmax_normalize(data_msi)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = FinalModel(
    #     hsi_channels=args.hsi_bands,
    #     msi_channels=args.msi_bands,
    #     hidden_dim=args.hidden_size,
    #     output_dim=128,
    #     delta_param_init=args.bn_threshold,
    #     num_classes=args.num_class,
    #     spatial_dim=args.patch_size,
    #     num_bands=args.hsi_bands,
    #     bn_threshold=args.bn_threshold,
    #     num_parallel=args.num_parallel,
    #     num_reslayer=args.num_reslayer
    # ).to(device)

    model =  FinalModel(args.hsi_bands, args.msi_bands, args.hidden_size, Bottleneck, args.num_parallel, args.num_reslayer,
                args.num_class, args.bn_threshold,spatial_dim=7,num_bands=144, hidden_dim=256,output_dim=256,delta_param_init=0.01, num_classes2=16,num_groups =130 ,
    group_size =11).eval().to(device)

    model.load_state_dict(torch.load("E:\code2024\spa and spe\HSI-MSI（data public）\models\model10e-65.pth"))

    margin = (args.patch_size - 1) // 2
    data_hsi = addZeroPadding(data_hsi, margin)
    data_msi = addZeroPadding(data_msi, margin)
    data_gt = np.pad(data_gt, ((margin, margin), (margin, margin)), 'constant', constant_values=(0, 0))


    idx, idy = np.where(data_gt != 0)
    labelss = np.array([0])

    batch = 100
    num = 10
    total_batch = int(len(idx)/batch +1)
    print ('Total batch number is :', total_batch)


    for j in range(total_batch):
        if batch * (j + 1) > len(idx):
            num_cat = len(idx) - batch * j
        else:
            num_cat = batch

        tmphsi = np.array([data_hsi[idx[j * batch + i] - margin:idx[j * batch + i] + margin + 1,
                                   idy[j * batch + i] - margin:idy[j * batch + i] + margin + 1, :] for i in range(num_cat)])
        tmpmsi = np.array([data_msi[idx[j * batch + i] - margin:idx[j * batch + i] + margin + 1,
                                   idy[j * batch + i] - margin:idy[j * batch + i] + margin + 1, :] for i in range(num_cat)])
        tmphsi1 = torch.FloatTensor(tmphsi.transpose(0, 3, 1, 2)).to(device)
        tmpmsi = torch.FloatTensor(tmpmsi.transpose(0, 3, 1, 2)).to(device)
        tmphsi2 = torch.FloatTensor(tmphsi.transpose(0, 3, 1, 2)).to(device)  # 根据实际需要修改

        # 执行模型推理
        with torch.no_grad():
            prediction = model.forward(tmphsi1, tmphsi2,tmpmsi)  # 传递 X_train2 作为第三个参数

        labelss = np.hstack([labelss, np.argmax(prediction.detach().cpu().numpy(), axis=1)])


    labelss = np.delete(labelss, [0])
    new_map = np.zeros((height, width))
    for i in range(len(idx)):
        new_map[idx[i] - margin, idy[i] - margin] = labelss[i] + 1

    scio.savemat(r"E:\code2024\spa and spe\HSI-MSI（data public）\results\result10e-61.mat", {'output': new_map})
    print('Finish!!!')
    data = scipy.io.loadmat(r"E:\code2024\spa and spe\HSI-MSI（data public）\results\result10e-61.mat")
    image_data = data['output']
    # plt.imshow(image_data, cmap='jet')
    # plt.colorbar()
    # plt.title('Classification Result')
    # plt.show()
    #


    # 加载数据
    pred_data = scipy.io.loadmat(r"E:\code2024\spa and spe\HSI-MSI（data public）\results\result10e-61.mat")
    true_data = scio.loadmat(r"E:\code2024\spa and spe\HSI-MSI（data public）\data\Houston2018\Sample.mat")
    # 获取预测和真实标签数据
    predicted = pred_data['output'].flatten()  # 确保与您文件中的变量名匹配
    true_labels = true_data['label'].flatten()  # 确保与您文件中的变量名匹配
    idx = np.where(true_labels > 0)
    true_labels = true_labels[idx]
    predicted = predicted[idx]
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted)

    # 计算精确度、召回率、F1分数和支持数
    precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, predicted)

    # 创建数据框架
    df = pd.DataFrame({
        'Class': np.arange(1, len(precision) + 1),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    })

    # 计算总体精度（OA）
    oa = accuracy_score(true_labels, predicted)

    # 计算平均精度（AA）
    aa = np.mean(recall)  # 使用召回率来计算AA

    # 计算Kappa系数
    kappa = cohen_kappa_score(true_labels, predicted)

    # 输出结果
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("Overall Accuracy (OA): {:.4f}".format(oa))
    print("Average Accuracy (AA): {:.4f}".format(aa))
    print("Kappa Coefficient: {:.4f}".format(kappa))


if __name__ == '__main__':
    main()


