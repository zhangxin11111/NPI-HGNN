import argparse
from src.npi_hgnn.model_classes import Model_1,Model_2,Model_3
from src.npi_hgnn.dataset_classes import NcRNA_Protein_Subgraph
import torch
from torch_geometric.data import DataLoader
def parse_args():
    parser = argparse.ArgumentParser(description="predict.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN3265", help='dataset name')
    parser.add_argument('--batchSize', default=32, type=int, help='batch size')
    parser.add_argument('--num_bases', default=2, type=int, help='Number of bases used for basis-decomposition')
    parser.add_argument('--num_relations', default=3, type=int, help='Number of edges')
    parser.add_argument('--model_code', default=1, type=int, help='model code')
    parser.add_argument('--cuda_code', default=0, type=int, help='cuda code')
    parser.add_argument('--num', default=3, type=int, help='文件名')
    return parser.parse_args()
def predict(model, loader, device):
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
    return TP, FN, TN, FP, pred
def Accuracy_Precision_Sensitivity_Specificity_MCC(model, loader, device):
    model.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    #pred = []
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
    print('predict case study start\n')
    args = parse_args()
    projectName = args.dataset
    print(args)
    case_study_path = f'../../data/{projectName}/case_study'
    case_study_predict_path=f'{case_study_path}/predict_dataset'
    predict_dataset = NcRNA_Protein_Subgraph(case_study_predict_path)
    predict_loader = DataLoader(predict_dataset, batch_size=args.batchSize)
    device = torch.device(f"cuda:{args.cuda_code}" if torch.cuda.is_available() else "cpu")
    if args.model_code==3:
        model = globals()[f'Model_{args.model_code}'](predict_dataset.num_node_features, 2).to(device)
    else:
        model = globals()[f'Model_{args.model_code}'](predict_dataset.num_node_features, args.num_relations,args.num_bases, 2).to(device)
    # Model storage address
    model_saving_path = f'{case_study_path}/model'
    model_saving_path=model_saving_path+f'/model_{args.model_code}.pth'
    model.load_state_dict(torch.load(model_saving_path))
    TP, FN, TN, FP, pred = predict(model,predict_loader,device)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    pred = sorted(pred, key=lambda x: x[3],reverse=True)
    output = 'case study dataset: Accuracy: %d, TP: %d, FN: %d, TN: %d, FP: %d' % (Accuracy,TP, FN, TN, FP)
    print(output)
    with open(f'./predict_{projectName}{args.num}.txt','w') as f:
        for i in pred:
            f.write(f'{i[0]},{i[1]},{i[2]},{i[3]}\n') # ncRNA protein true_label pos_sample_prob

    print('predict case study end\n')