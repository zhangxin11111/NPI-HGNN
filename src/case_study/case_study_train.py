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
    now_time = time.localtime()
    time_format = '%Y-%m-%d %H:%M:%S'
    time_put = time.strftime(time_format,now_time)
    log_file = open(path,'a')
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
def train(model,data_loader):
    model.train()
    loss_all = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def Accuracy_Precision_Sensitivity_Specificity_MCC(model, loader, device,log_path):
    model.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
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
    args = parse_args()
    projectName = 'NPHN7317'
    print(args)
    case_study_path = f'../../data/{projectName}/case_study'
    case_study_train_path=f'{case_study_path}/train_dataset'
    case_study_val_path=f'{case_study_path}/val_dataset'
    train_dataset = NcRNA_Protein_Subgraph(case_study_train_path)
    val_dataset = NcRNA_Protein_Subgraph(case_study_val_path)
    print('shuffle dataset\n')
    train_dataset = train_dataset.shuffle()
    val_dataset = val_dataset.shuffle()
    device = torch.device(f"cuda:{args.cuda_code}" if torch.cuda.is_available() else "cpu")
    log_saving_path=f'{case_study_path}/log'
    print(log_saving_path)
    if not osp.exists(log_saving_path):
        os.makedirs(log_saving_path)
    num_of_epoch = args.epochNumber
    LR = args.initialLearningRate
    L2_weight_decay = args.l2WeightDecay
    log_path = log_saving_path + f'/model_{args.model_code}.txt'
    if (os.path.exists(log_path)):
        os.remove(log_path)
    write_log(log_path,f'dataset：{projectName}\n')
    write_log(log_path,f'training dataset path: {case_study_train_path}\n')
    write_log(log_path,f'number of eopch ：{num_of_epoch}\n')
    write_log(log_path,f'learn rate：initial = {LR}，whenever loss increases, multiply by 0.95\n')
    write_log(log_path,f'L2 weight decay = {L2_weight_decay}\n')
    start_time = time.time()
    model_saving_path=f'{case_study_path}/model'
    if not osp.exists(model_saving_path):
        os.makedirs(model_saving_path)
    if args.model_code==1:
        model = globals()[f'Model_{args.model_code}'](train_dataset.num_node_features, 2).to(device)
    else:
        model = globals()[f'Model_{args.model_code}'](train_dataset.num_node_features, args.num_relations,args.num_bases, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    print(f'number of samples in training dataset：{str(len(train_dataset))}\n')
    write_log(log_path,f'number of samples in training dataset：{str(len(train_dataset))}\n')
    print('training dataset')
    dataset_analysis(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batchSize)
    print(f'number of samples in val dataset：{str(len(val_dataset))}\n')
    write_log(log_path,f'number of samples in cal dataset：{str(len(val_dataset))}\n')
    print('val dataset')
    dataset_analysis(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batchSize)
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
        if loss > loss_last:
            scheduler.step()
        loss_last = loss
        Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model,
                                                                                                            train_loader,
                                                                                                            device,
                                                                                                            log_path)
        output = 'Epoch: {:03d}, train dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
            epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
        print(output)

        Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model,val_loader,device,log_path)
        output = 'Epoch: {:03d}, val dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
            epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
        print(output)
        write_log(log_path, output + '\n')
        if MCC > MCC_max:
            MCC_max = MCC
            epoch_MCC_max = epoch + 1
            ACC_MCC_max = Accuracy
            Pre_MCC_max = Precision
            Sen_MCC_max = Sensitivity
            Spe_MCC_max = Specificity
            early_stop = 0
            network_model_path = model_saving_path + f'/{epoch + 1}'
            torch.save(model.state_dict(), network_model_path)
        else:
            early_stop += 1
        if early_stop > 10:
            break

    output = 'Best performance in val: Epoch: {:03d} , Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
        epoch_MCC_max, ACC_MCC_max, Pre_MCC_max, Sen_MCC_max, Spe_MCC_max, MCC_max)
    print(output)
    write_log(log_path,output + '\n')
    end_time = time.time()
    print('Time consuming:', end_time - start_time)
    write_log(log_path,'Time consuming:' + str(end_time - start_time) + '\n')
    print('\nexit\n')
    from sklearn.model_selection import cross_val_score