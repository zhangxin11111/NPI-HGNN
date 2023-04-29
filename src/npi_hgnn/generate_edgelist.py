# coding:utf-8
import argparse
import os.path as osp
import os
from src.npi_hgnn.methods import read_rpi,random_negative_sampling,write_interactor,generate_pre_dataset_path
from src.npi_hgnn.methods import fire_negative_sampling,read_sequence_file,reliable_negative_sampling,read_rrsn,case_study_sampling
from sklearn.model_selection import StratifiedKFold,train_test_split
import numpy as np
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Negative sample selection and partitioning the dataset.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
    # 0：RANDOM 1：FIRE 2：RelNegNPI 3、Rel&Ran
    parser.add_argument('--samplingType',default=3,type=int, help='sampling type')
    parser.add_argument('--num_fold', default=5,type=int, help='how num of fold is this')
    return parser.parse_args()

def generate_training_and_testing(set_interaction, set_negativeInteraction,path_cross_valid,num_fold, rna_name_set,protein_name_set):
    list_set_interactionKey=[]
    list_set_negativeInteractionKey=[]
    for i in range(num_fold):
        list_set_interactionKey.append(set())
        list_set_negativeInteractionKey.append(set())
    count = 0
    while len(set_interaction) > 0:
        list_set_interactionKey[count % num_fold].add(set_interaction.pop())
        count += 1
    count = 0
    while len(set_negativeInteraction) > 0:
        list_set_negativeInteractionKey[count % num_fold].add(set_negativeInteraction.pop())
        count += 1
    for i in range(num_fold):
        pos_train_edges = set()
        neg_train_edges = set()
        pos_test_edges = set()
        neg_test_edges = set()

        for j in range(num_fold):
            if i == j:
                pos_test_edges.update(list_set_interactionKey[j])
                neg_test_edges.update(list_set_negativeInteractionKey[j])
            else:
                pos_train_edges.update(list_set_interactionKey[j])
                neg_train_edges.update(list_set_negativeInteractionKey[j])
        if not osp.exists(path_cross_valid + f'/dataset_{i}'):
            os.makedirs(path_cross_valid + f'/dataset_{i}')
        write_interactor(pos_test_edges, path_cross_valid + f'/dataset_{i}/pos_test_edges')
        write_interactor(neg_test_edges, path_cross_valid + f'/dataset_{i}/neg_test_edges')
        #从训练集中抽出一部分作为验证集
        pos_train_edges=np.array(list(pos_train_edges))
        pos_train_edges, pos_val_edges = train_test_split(pos_train_edges,test_size=0.1, random_state=1337)
        neg_train_edges=np.array(list(neg_train_edges))
        neg_train_edges, neg_val_edges = train_test_split(neg_train_edges,test_size=0.1, random_state=1337)
        write_interactor(pos_train_edges, path_cross_valid + f'/dataset_{i}/pos_train_edges')
        write_interactor(neg_train_edges, path_cross_valid + f'/dataset_{i}/neg_train_edges')
        write_interactor(pos_val_edges, path_cross_valid + f'/dataset_{i}/pos_val_edges')
        write_interactor(neg_val_edges, path_cross_valid + f'/dataset_{i}/neg_val_edges')
        # catnot_use = set()
        # catnot_use.update(pos_train_edges)
        # catnot_use.update(neg_train_edges)
        # catnot_use.update(pos_test_edges)


def get_k_fold_data(k, data):
    X, y = data[:, :], data[:, -1]
    sfolder = StratifiedKFold(n_splits=k, shuffle=True)

    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for train, test in sfolder.split(X, y):
        train_data.append(X[train])
        test_data.append(X[test])
        train_label.append(y[train])
        test_label.append(y[test])
    return train_data, test_data
def output_dict_file(dict,path):
    output_file = open(path, mode='w')
    for item in dict.items():
        output_file.write(item[0] + '\n')
    output_file.close()
if __name__ == '__main__':
    print('start partition dataset\n')
    args = parse_args()
    input_path=f'../../data/{args.dataset}/processed_database_data/protein_sequence.fasta'
    name_list, sequence_list = read_sequence_file(input_path)
    dict_protein_name_id=dict(zip(name_list,range(len(name_list))))
    input_path=f'../../data/{args.dataset}/processed_database_data/ncRNA_sequence.fasta'
    name_list, sequence_list = read_sequence_file(input_path)
    dict_rna_name_id=dict(zip(name_list,range(len(name_list))))
    rpi_path=f'../../data/{args.dataset}/processed_database_data/{args.dataset}.xlsx'
    positive_samples,rna_name_set,protein_name_set=read_rpi(rpi_path)
    node_name_set=rna_name_set.union(protein_name_set)
    pre_dataset_path= generate_pre_dataset_path(args.dataset,args.samplingType)
    if not osp.exists(pre_dataset_path):
        os.makedirs(pre_dataset_path)
    with open(pre_dataset_path+f'/all_node_name',mode='w') as f:
        for item in node_name_set:
            f.write(item+'\n')
    pp_swscore_matrix = pd.read_csv(f'../../data/{args.dataset}/source_database_data/PPSM/ppsm.txt',header=None).values
    rr_swscore_matrix = pd.read_csv(f'../../data/{args.dataset}/source_database_data/RRSM/rrsm.txt',header=None).values
    if args.samplingType==0:
        set_interaction = [(triplet[0], triplet[2]) for triplet in positive_samples.values]
        set_interaction = set(set_interaction)
        set_negativeInteraction = random_negative_sampling(set_interaction, rna_name_set, protein_name_set, len(set_interaction))

    elif  args.samplingType==1:
        set_interaction=positive_samples[['source','target']].values.tolist()
        Positives, Negatives = fire_negative_sampling(set_interaction, rna_name_set,protein_name_set, pp_swscore_matrix,dict_protein_name_id,len(set_interaction))
        set_interaction = set(Positives)
        set_negativeInteraction = set(Negatives)
    elif  args.samplingType==2:
        rrsn,rna_names=read_rrsn(f'../../data/{args.dataset}/processed_database_data/{args.dataset}_RRI.xlsx')
        set_interaction=positive_samples[['source', 'target']].values.tolist()
        Positives, Negatives = reliable_negative_sampling(set_interaction, rna_name_set,protein_name_set, pp_swscore_matrix,dict_protein_name_id,rr_swscore_matrix,dict_rna_name_id,0.5,len(set_interaction))
        set_interaction = set(Positives)
        set_negativeInteraction = set(Negatives)
    elif args.samplingType == 3:
        rrsn,rna_names=read_rrsn(f'../../data/{args.dataset}/processed_database_data/{args.dataset}_RRI.xlsx')
        set_interaction=positive_samples[['source', 'target']].values.tolist()
        Positives, reliable_negatives = reliable_negative_sampling(set_interaction, rna_name_set,protein_name_set, pp_swscore_matrix,dict_protein_name_id,rr_swscore_matrix,dict_rna_name_id,0.5,len(set_interaction)//2)
        set_interaction = set(Positives)
        set_negativeInteraction = set(reliable_negatives)
        random_negatives = random_negative_sampling(set_interaction, rna_name_set, protein_name_set,len(set_interaction)//2)
        set_negativeInteraction.update(random_negatives)
    write_interactor(set_interaction, pre_dataset_path + '/all_postitive_edges')
    write_interactor(set_negativeInteraction, pre_dataset_path + '/all_negative_edges')

    set_interaction = [(triplet[0], triplet[1]) for triplet in set_interaction]
    generate_training_and_testing(set_interaction,set_negativeInteraction,pre_dataset_path,args.num_fold, rna_name_set,protein_name_set)
    print('partition dataset end\n')