import argparse
from src.npi_hgnn.dataset_classes import NcRNA_Protein_Subgraph
import torch
import os.path as osp
import os
import time
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from src.npi_hgnn.model_classes import Model_1,Model_2
from src.npi_hgnn.methods import generate_dataset_path,generate_log_path,generate_model_path
def generate_result_path(dataset,nodeVecType,subgraph_type,samplingType):
    path=f'../../data/result/{dataset}'
    if subgraph_type==0:
        path=f'{path}/subgraph3'
    elif subgraph_type==1:
        path=f'{path}/subgraph2'
    elif subgraph_type==2:
        path=f'{path}/subgraph1'
    if nodeVecType==0:
        path=f'{path}/frequency'
    elif nodeVecType==1:
        path=f'{path}/pyfeat'
    elif nodeVecType==2:
        path=f'{path}/no_sequence'
    elif nodeVecType==3:
        path=f'{path}/only_frequency'
    if samplingType==0:
        path=f'{path}/random'
    elif samplingType==1:
        path=f'{path}/fire'
    elif samplingType==2:
        path=f'{path}/reliable'
    elif samplingType==3:
        path=f'{path}/random_reliable'
    return path
def generate_pred_path(dataset,nodeVecType,subgraph_type,samplingType):
    path=f'../../data/pred/{dataset}'
    if subgraph_type==0:
        path=f'{path}/subgraph3'
    elif subgraph_type==1:
        path=f'{path}/subgraph2'
    elif subgraph_type==2:
        path=f'{path}/subgraph1'
    if nodeVecType==0:
        path=f'{path}/frequency'
    elif nodeVecType==1:
        path=f'{path}/pyfeat'
    elif nodeVecType==2:
        path=f'{path}/no_sequence'
    elif nodeVecType==3:
        path=f'{path}/only_frequency'
    if samplingType==0:
        path=f'{path}/random'
    elif samplingType==1:
        path=f'{path}/fire'
    elif samplingType==2:
        path=f'{path}/reliable'
    elif samplingType==3:
        path=f'{path}/random_reliable'
    return path
def parse_args():
    parser = argparse.ArgumentParser(description="train.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
    # 0:kmer || n2v 1: pyfeat || n2v 2:n2v
    parser.add_argument('--nodeVecType', default=0,type=int, help='node vector type')
    # 0 subgraph3 ；1 subgraph2 ；2 subgraph1
    parser.add_argument('--subgraph_type', default=1, type=int, help='if use complete subgraph')
    # 0：RANDOM 1：FIRE 2：RelNegNPI 3、Rel&Ran
    parser.add_argument('--samplingType', default=3,type=int, help='negative sampling type')
    parser.add_argument('--fold', default=0,type=int,help='which fold is this')
    parser.add_argument('--epochNumber', default=100, type=int, help='number of training epoch')
    parser.add_argument('--initialLearningRate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--l2WeightDecay', default=0.001, type=float, help='L2 weight')
    parser.add_argument('--batchSize', default=32, type=int, help='batch size')
    parser.add_argument('--num_bases', default=2, type=int, help='Number of bases used for basis-decomposition')
    parser.add_argument('--num_relations', default=3, type=int, help='Number of edges')
    parser.add_argument('--model_code', default=1, type=int, help='model code')
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
def write_result(path,value):
    log_file = open(path,'a')
    write_value = f'[{value}],\n'
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
def Accuracy_Precision_Sensitivity_Specificity_MCC_Pred(model, loader, device):
    model.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    pred = []

    for data in loader:
        data = data.to(device)
        pred_label = model(data).max(dim=1)[1]
        for index in range(len(pred_label)):
            if pred_label[index] == 1 and data.y[index] == 1:
                TP += 1
            elif pred_label[index] == 1 and data.y[index] == 0:
                FP += 1
            elif pred_label[index] == 0 and data.y[index] == 1:
                FN += 1
            else:
                TN += 1
        pred_prob=torch.exp(model(data))[:,1].tolist()
        pred.extend([(value[0], value[1],data.y.tolist()[i], pred_prob[i]) for i, value in enumerate(data.target_link)])
    output = 'TP: %d, FN: %d, TN: %d, FP: %d' % (TP, FN, TN, FP)
    print(output)
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
    return Accuracy, Precision, Sensitivity, Specificity, MCC,pred
if __name__ == "__main__":
    args = parse_args()
    print(args)
    dataset_train_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'train')
    dataset_test_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'test')
    dataset_val_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'val')
    train_dataset = NcRNA_Protein_Subgraph(dataset_train_path)
    test_dataset = NcRNA_Protein_Subgraph(dataset_test_path)
    val_dataset = NcRNA_Protein_Subgraph(dataset_val_path)
    print('shuffle dataset\n')
    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()
    val_dataset = val_dataset.shuffle()
    device = torch.device(f"cuda:{args.cuda_code}" if torch.cuda.is_available() else "cpu")
    log_saving_path=generate_log_path(args.dataset,args.model_code,args.nodeVecType,args.subgraph_type,args.samplingType)
    print(log_saving_path)
    if not osp.exists(log_saving_path):
        os.makedirs(log_saving_path)
    num_of_epoch = args.epochNumber
    LR = args.initialLearningRate
    L2_weight_decay = args.l2WeightDecay
    log_path = log_saving_path + f'/fold_{args.fold}.txt'
    if (os.path.exists(log_path)):
        os.remove(log_path)
    write_log(log_path,f'dataset：{args.dataset}\n')
    write_log(log_path,f'training dataset path: {dataset_train_path}\n')
    write_log(log_path,f'testing dataset path: {dataset_test_path}\n')
    write_log(log_path,f'number of eopch ：{num_of_epoch}\n')
    write_log(log_path,f'learn rate：initial = {LR}，whenever loss increases, multiply by 0.95\n')
    write_log(log_path,f'L2 weight decay = {L2_weight_decay}\n')
    start_time = time.time()
    model_saving_path=generate_model_path(args.dataset,args.model_code,args.nodeVecType,args.subgraph_type,args.fold,args.samplingType)
    if not osp.exists(model_saving_path):
        os.makedirs(model_saving_path)
    if(train_dataset.num_node_features != test_dataset.num_node_features):
        raise Exception('The node feature dimensions of training set and test set are inconsistent')
    model = globals()[f'Model_{args.model_code}'](train_dataset.num_node_features, args.num_relations, args.num_bases,2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    print(f'number of samples in training dataset：{str(len(train_dataset))}\n')
    print(f'number of samples in testing dataset：{str(len(test_dataset))}\n')
    write_log(log_path,f'number of samples in training dataset：{str(len(train_dataset))}\n')
    write_log(log_path, f'number of samples in testing dataset：{str(len(test_dataset))}\n')
    print('training dataset')
    dataset_analysis(train_dataset)
    print('testing dataset')
    dataset_analysis(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batchSize)
    test_loader = DataLoader(test_dataset, batch_size=args.batchSize)
    val_loader = DataLoader(val_dataset, batch_size=args.batchSize)
    MCC_max = -1
    epoch_MCC_max = 0
    ACC_MCC_max = 0
    Pre_MCC_max = 0
    Sen_MCC_max = 0
    Spe_MCC_max = 0
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
        write_log(log_path, output + '\n')

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
    write_log(log_path, output + '\n')

    model.load_state_dict(torch.load(model_saving_path + f'/{epoch_MCC_max}'))
    Accuracy, Precision, Sensitivity, Specificity, MCC,pred = Accuracy_Precision_Sensitivity_Specificity_MCC_Pred(model,test_loader,device)
    output = 'test dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
         Accuracy, Precision, Sensitivity, Specificity, MCC)
    print(output)
    path = generate_result_path(args.dataset,args.nodeVecType,args.subgraph_type,args.samplingType)
    if not osp.exists(path):
        os.makedirs(path)
    write_result( f'{path}/{args.model_code}.txt',f'{Accuracy},{Precision},{Sensitivity},{Specificity},{MCC}')

    path = generate_pred_path(args.dataset,args.nodeVecType,args.subgraph_type,args.samplingType)
    if not osp.exists(path):
        os.makedirs(path)
    with open(f'{path}/{args.dataset}_model{args.model_code}.txt','a') as f:
        for i in pred:
            f.write(f'{i[0]}\t{i[1]}\t{i[2]}\t{i[3]}\n')
    end_time = time.time()
    print('Time consuming:', end_time - start_time)
    write_log(log_path,'Time consuming:' + str(end_time - start_time) + '\n')
    print('\nexit\n')