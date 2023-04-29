import argparse
from src.npi_hgnn.methods import read_rpin,read_ppin,read_rrsn,generate_dataset_path,generate_pre_dataset_path,generate_node_path
from src.npi_hgnn.dataset_classes import NcRNA_Protein_Subgraph
import os.path as osp
import os
import pandas as pd
import shutil
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
    # 0:kmer || n2v 1: pyfeat || n2v 2:n2v
    parser.add_argument('--nodeVecType',type=int, default=0, help='node vector type')
    # 0：RANDOM 1：FIRE 2：RelNegNPI 3、Rel&Ran
    parser.add_argument('--samplingType',type=int, default=3, help='negative sampling type')
    # 0 subgraph3 ；1 subgraph2 ；2 subgraph1
    parser.add_argument('--subgraph_type', default=1, type=int, help='type of subgraph')
    parser.add_argument('--fold',type=int, default=0,help='which fold is this')
    return parser.parse_args()
def read_interaction(path):
    interaction = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            interaction.add((arr[0], arr[1]))
    return interaction
if __name__ == "__main__":
    print('start generate pytorch dataset\n')
    args=parse_args()
    pre_dataset_path=generate_pre_dataset_path(args.dataset,args.samplingType)
    node_path=generate_node_path(args.dataset,args.samplingType,args.nodeVecType)
    rpin, rna_names1, protein_names1 = read_rpin(f'{pre_dataset_path}/dataset_{args.fold}/pos_train_edges')
    ppin,_=read_ppin(f'../../data/{args.dataset}/processed_database_data/{args.dataset}_PPI.xlsx')
    rrsn,_=read_rrsn(f'../../data/{args.dataset}/processed_database_data/{args.dataset}_RRI.xlsx')
    rna_vec_path = f'{node_path}/node_vec/{args.fold}/rna_vec.txt'
    protein_vec_path = f'{node_path}/node_vec/{args.fold}/protein_vec.txt'
    node_vecs = pd.read_csv(rna_vec_path, header=None).append(pd.read_csv(protein_vec_path, header=None)).reset_index(drop=True)

    dict_node_name_vec = dict(zip(node_vecs.values[:,0],node_vecs.values[:,1:]) )
    dataset_train_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'train')
    if not osp.exists(dataset_train_path):
        os.makedirs(dataset_train_path)
    else:
        shutil.rmtree(dataset_train_path,True)
        os.makedirs(dataset_train_path)
    path_pos_train =  f'{pre_dataset_path}/dataset_{args.fold}/pos_train_edges'
    path_neg_train =f'{pre_dataset_path}/dataset_{args.fold}/neg_train_edges'
    pos_train = read_interaction(path_pos_train)
    neg_train = read_interaction(path_neg_train)
    train_interactions=[]
    train_interactions.extend(pos_train)
    num_pos_train = len(train_interactions)
    train_interactions.extend(neg_train)
    num_neg_train = len(train_interactions) - num_pos_train
    y = np.ones(num_pos_train).tolist()
    y.extend(np.zeros(num_neg_train).tolist())
    train_dataset = NcRNA_Protein_Subgraph(dataset_train_path,rpin,ppin,rrsn,dict_node_name_vec,train_interactions,y,args.subgraph_type)
    dataset_test_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'test')
    if not osp.exists(dataset_test_path):
        os.makedirs(dataset_test_path)
    else:
        shutil.rmtree(dataset_test_path,True)
        os.makedirs(dataset_test_path)
    pre_dataset_path=generate_pre_dataset_path(args.dataset,args.samplingType)
    path_neg_test =f'{pre_dataset_path}/dataset_{args.fold}/neg_test_edges'
    path_pos_test =  f'{pre_dataset_path}/dataset_{args.fold}/pos_test_edges'
    pos_test = read_interaction(path_pos_test)
    neg_test= read_interaction(path_neg_test)
    test_interactions=[]
    test_interactions.extend(pos_test)
    num_pos_test=len(test_interactions)
    test_interactions.extend(neg_test)
    num_neg_test = len(test_interactions)-num_pos_test
    y=np.ones(num_pos_test).tolist()
    y.extend(np.zeros(num_neg_test).tolist())
    test_dataset = NcRNA_Protein_Subgraph(dataset_test_path, rpin, ppin, rrsn,dict_node_name_vec,test_interactions,y,args.subgraph_type)

    dataset_val_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'val')
    if not osp.exists(dataset_val_path):
        os.makedirs(dataset_val_path)
    else:
        shutil.rmtree(dataset_val_path,True)
        os.makedirs(dataset_val_path)
    pre_dataset_path=generate_pre_dataset_path(args.dataset,args.samplingType)
    path_neg_val =f'{pre_dataset_path}/dataset_{args.fold}/neg_val_edges'
    path_pos_val =  f'{pre_dataset_path}/dataset_{args.fold}/pos_val_edges'
    pos_val = read_interaction(path_pos_val)
    neg_val= read_interaction(path_neg_val)
    val_interactions=[]
    val_interactions.extend(pos_val)
    num_pos_val=len(val_interactions)
    val_interactions.extend(neg_val)
    num_neg_val = len(val_interactions)-num_pos_val
    y=np.ones(num_pos_val).tolist()
    y.extend(np.zeros(num_neg_val).tolist())
    val_dataset = NcRNA_Protein_Subgraph(dataset_val_path, rpin, ppin, rrsn,dict_node_name_vec,val_interactions,y,args.subgraph_type)

    print('generate pytorch dataset end\n')