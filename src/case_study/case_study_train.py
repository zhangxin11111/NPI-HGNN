import argparse
from src.npi_hgnn.dataset_classes import NcRNA_Protein_Subgraph
import torch
import os.path as osp
import os
import time
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from src.npi_hgnn.model_classes import Model_1,Model_2
def parse_args():
    parser = argparse.ArgumentParser(description="train.")

    parser.add_argument('--epochNumber', default=100, type=int, help='number of training epoch')
    parser.add_argument('--initialLearningRate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--l2WeightDecay', default=0.001, type=float, help='L2 weight')
    parser.add_argument('--batchSize', default=32, type=int, help='batch size')
    parser.add_argument('--num_bases', default=2, type=int, help='Number of bases used for basis-decomposition')
    parser.add_argument('--num_relations', default=3, type=int, help='Number of edges')
    parser.add_argument('--model_code', default=2, type=int, help='model code') # 1 2
    parser.add_argument('--cuda_code', default=0, type=int, help='cuda code')
    parser.add_argument('--droupout_ratio', default=0.5, type=float, help='droupout_ratio')
    parser.add_argument('--gamma', default=0.95, type=float, help='gamma')
    return parser.parse_args()
def write_log(path,value):
    now_time = time.localtime() #获取当前日期和时间
    time_format = '%Y-%m-%d %H:%M:%S' #指定日期和时间格式
    time_put = time.strftime(time_format,now_time) #格式化时间，时间变成YYYY-MM-DD HH:MI:SS
    log_file = open(path,'a') #这里用追加模式，如果文件不存在的话会自动创建
    write_value = '%s %s' %(time_put,value)
    log_file.write(write_value)
    log_file.close()
def dataset_analysis(dataset):
    dict_label_dataNumber = {}
    for data in dataset:
        label = int(data.y)
        if label not in dict_label_dataNumber:
            dict_label_dataNumber[label] = 1
        else:
            dict_label_dataNumber[label] = dict_label_dataNumber[label] + 1
    print(dict_label_dataNumber)
def train(model,train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        #assert torch.isnan(loss).sum() == 0, print(loss)
        loss.backward() #计算梯度
        loss_all += data.num_graphs * loss.item()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
        optimizer.step() #更新参数
    return loss_all / len(train_dataset)


def Accuracy_Precision_Sensitivity_Specificity_MCC(model, loader, device,log_path):
    model.eval()
    TP = 0 # TP：被模型预测为正类的正样本
    TN = 0 # TN：被模型预测为负类的负样本
    FP = 0 # FP：被模型预测为正类的负样本
    FN = 0 # FN：被模型预测为负类的正样本
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        for index in range(len(pred)):
            if pred[index] == 1 and data.y[index] == 1:
                TP += 1
            elif pred[index] == 1 and data.y[index] == 0:
                FP += 1
            elif pred[index] == 0 and data.y[index] == 1:
                FN += 1
            else:
                TN += 1
    output = 'TP: %d, FN: %d, TN: %d, FP: %d' % (TP, FN, TN, FP)
    print(output)
    write_log(log_path,output + '\n')
    if (TP + TN + FP + FN) != 0:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        Accuracy = 0
    if (TP + FP) != 0:
        Precision = (TP) / (TP + FP)
    else:
        Precision = 0
    if (TP + FN) != 0:
        Sensitivity = (TP) / (TP + FN)
    else:
        Sensitivity = 0
    if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) != 0:
        MCC = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    else:
        MCC = 0
    if (FP + TN) != 0:
        Specificity = TN / (FP + TN)
    else:
        Specificity = 0
    return Accuracy, Precision, Sensitivity, Specificity, MCC
if __name__ == "__main__":
    #参数
    args = parse_args()
    projectName = 'NPHN7317'
    print(args)
    case_study_path = f'../../data/{projectName}/case_study'
    case_study_train_path=f'{case_study_path}/train_dataset'
    #生成负样本随机的测试集
    train_dataset = NcRNA_Protein_Subgraph(case_study_train_path)
    # 打乱数据集
    print('shuffle dataset\n')
    train_dataset = train_dataset.shuffle()
    #选择CPU或CUDA
    device = torch.device(f"cuda:{args.cuda_code}" if torch.cuda.is_available() else "cpu")
    # 准备日志
    log_saving_path=f'{case_study_path}/log'
    print(log_saving_path)
    if not osp.exists(log_saving_path):
        print(f'创建日志文件夹：{log_saving_path}')
        os.makedirs(log_saving_path)
     # 迭代次数
    num_of_epoch = args.epochNumber
    # 学习率
    LR = args.initialLearningRate
    # L2正则化系数
    L2_weight_decay = args.l2WeightDecay
    # 日志基本信息写入
    log_path = log_saving_path + f'/model_{args.model_code}.txt'
    if (os.path.exists(log_path)):
        os.remove(log_path)
    write_log(log_path,f'dataset：{projectName}\n')
    write_log(log_path,f'training dataset path: {case_study_train_path}\n')
    write_log(log_path,f'number of eopch ：{num_of_epoch}\n')
    write_log(log_path,f'learn rate：initial = {LR}，whenever loss increases, multiply by 0.95\n')
    write_log(log_path,f'L2 weight decay = {L2_weight_decay}\n')
    # 记录启起始时间
    start_time = time.time()
    model_saving_path=f'{case_study_path}/model'
    if not osp.exists(model_saving_path):
        print(f'创建保存模型文件夹：{model_saving_path}')
        os.makedirs(model_saving_path)
    #创建模型
    if args.model_code==1:
        model = globals()[f'Model_{args.model_code}'](train_dataset.num_node_features, 2).to(device)
    else:
        model = globals()[f'Model_{args.model_code}'](train_dataset.num_node_features, args.num_relations,args.num_bases, 2).to(device)
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    # 训练集
    print(f'number of samples in training dataset：{str(len(train_dataset))}\n')
    write_log(log_path,f'number of samples in training dataset：{str(len(train_dataset))}\n')
    print('training dataset')
    dataset_analysis(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batchSize)
    MCC_max = -1
    epoch_MCC_max = 0
    ACC_MCC_max = 0
    Pre_MCC_max = 0
    Sen_MCC_max = 0
    Spe_MCC_max = 0
    # 训练开始
    loss_last = float('inf')
    early_stop=0
    for epoch in range(num_of_epoch):
        loss = train(model,train_loader)
        # loss增大时,降低学习率
        if loss > loss_last:
            scheduler.step()
        loss_last = loss
        # 训练中评价模型，监视训练过程中的模型变化, 并且写入文件
        if (epoch + 1) % 1 == 0 and epoch != num_of_epoch - 1:
            # 用Accuracy, Precision, Sensitivity, MCC评价模型
            Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, train_loader, device,log_path)
            output = 'Epoch: {:03d}, training dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
            print(output)
            write_log(log_path,output + '\n')
            if MCC > MCC_max:
                MCC_max = MCC
                epoch_MCC_max = epoch+1
                ACC_MCC_max = Accuracy
                Pre_MCC_max = Precision
                Sen_MCC_max = Sensitivity
                Spe_MCC_max = Specificity
                early_stop=0
                # 保存模型
                network_model_path = model_saving_path + f'/{epoch + 1}'
                torch.save(model.state_dict(), network_model_path)
            else:
                early_stop+=1
            if early_stop>20:
                break
    # 训练结束，评价模型，并且把结果写入文件
    Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, train_loader, device,log_path)
    output = 'Epoch: {:03d}, training dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
    print(output)
    write_log(log_path,output + '\n')
    if MCC > MCC_max:
        MCC_max = MCC
        epoch_MCC_max = args.epochNumber
        ACC_MCC_max = Accuracy
        Pre_MCC_max = Precision
        Sen_MCC_max = Sensitivity
        Spe_MCC_max = Specificity
        # 保存模型
        network_model_path = model_saving_path + f'/{epoch + 1}'
        torch.save(model.state_dict(), network_model_path)
    write_log(log_path,'\n')
    output = f'MCC最大的时候的性能：'
    print(output)
    write_log(log_path,output + '\n')
    output = f'epoch: {epoch_MCC_max}, ACC: {ACC_MCC_max}, Pre: {Pre_MCC_max}, Sen: {Sen_MCC_max}, Spe: {Spe_MCC_max}, MCC: {MCC_max}'
    print(output)
    write_log(log_path,output + '\n')
    # 完毕
    end_time = time.time()
    print('Time consuming:', end_time - start_time)
    write_log(log_path,'Time consuming:' + str(end_time - start_time) + '\n')
    print('\nexit\n')