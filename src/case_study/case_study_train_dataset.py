import argparse
from src.npi_hgnn.methods import read_ppin,read_rrsn,read_RPI,read_PPI,read_RRI,generate_n2v,read_sequence_file,read_node2vec_file,read_kmer_fre_file,read_rpin,read_all
from src.npi_hgnn.dataset_classes import NcRNA_Protein_Subgraph
import os.path as osp
import os
import pandas as pd
import shutil
import numpy as np
import networkx as nx
import gc
def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN3265", help='dataset name')
    # 0 subgraph3 ；1 subgraph2 ；2 subgraph1
    parser.add_argument('--subgraph_type', default=2, type=int, help='type of subgraph')
    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    return parser.parse_args()
def read_interaction(path):
    interaction = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            interaction.add((arr[0], arr[1]))
    return interaction
def generate_node_vec_with_fre(projectName,node_path):
    node_name_vec_dict=read_node2vec_file(f'{node_path}/node2vec/result.emb')
    #生成rna节点特征
    input_path_rna = f'../../data/{projectName}/processed_database_data/ncRNA_sequence.fasta'
    rna_name_list, rna_sequence_list = read_sequence_file(path=input_path_rna)
    rna_kmer_fre_path=f'../../data/{projectName}/frequency/rna_seq/result.emb'
    fre_name_list, fre_list=read_kmer_fre_file(rna_kmer_fre_path)
    fre_name_index_dict = {} #rna_name,id
    for index in range(len(fre_name_list)):
        fre_name_index_dict[fre_name_list[index]] = index
    if not osp.exists(f'{node_path}/node_vec'):
        os.makedirs(f'{node_path}/node_vec')
    with open(f'{node_path}/node_vec/rna_vec.txt',mode='w') as f:
        for rna_name in rna_name_list:
            node_vec=fre_list[fre_name_index_dict[rna_name]]
            for i in range(49):
                node_vec.append('0')
            node2vec=node_name_vec_dict[rna_name]
            node_vec.extend(node2vec)
            f.write(rna_name+','+','.join(node_vec)+'\n')
    input_path_protein = f'../../data/{projectName}/processed_database_data/protein_sequence.fasta'
    protein_name_list, protein_sequence_list = read_sequence_file(path=input_path_protein)
    protein_kmer_fre_path=f'../../data/{projectName}/frequency/protein_seq/result.emb'
    fre_name_list, fre_list=read_kmer_fre_file(protein_kmer_fre_path)
    fre_name_index_dict = {} #protein_name,id
    for index in range(len(fre_name_list)):
        fre_name_index_dict[fre_name_list[index]] = index
    with open(f'{node_path}/node_vec/protein_vec.txt',mode='w') as f:
        for protein_name in protein_name_list:
            node_vec=[]
            for i in range(64):
                node_vec.append('0')
            node_vec.extend(fre_list[fre_name_index_dict[protein_name]])
            node2vec=node_name_vec_dict[protein_name]
            node_vec.extend(node2vec)
            f.write(protein_name+','+','.join(node_vec)+'\n')

if __name__ == "__main__":
    args = parse_args()
    projectName = args.dataset
    # Generating node characteristics
    case_study_path=f'../../data/{projectName}/case_study'
    print('start generate node feature vector\n')
    node_names = []
    with open(f'{case_study_path}/all_node_name') as f:
        lines = f.readlines()
        for line in lines:
            node_names.append(line.strip())
    graph_path = f'{case_study_path}/case_study_train_pos_edges'
    if args.subgraph_type == 2:
        G, _, _ = read_rpin(graph_path)
    else:
        G, _, _ = read_all(graph_path, projectName)
    G.add_nodes_from(node_names)
    generate_n2v(G, f'{case_study_path}/node2vec', args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
    generate_node_vec_with_fre(projectName, case_study_path)
    del G
    gc.collect()
    print('generate node feature vector end\n')
    # Generating Dataset
    print('start generate case study dataset\n')
    rpin, _, _ = read_rpin(f'../../data/{projectName}/case_study/case_study_train_pos_edges')
    ppin,_=read_ppin(f'../../data/{projectName}/processed_database_data/{projectName}_PPI.xlsx')
    rrsn,_=read_rrsn(f'../../data/{projectName}/processed_database_data/{projectName}_RRI.xlsx')
    rna_vec_path = f'{case_study_path}/node_vec/rna_vec.txt'
    protein_vec_path = f'{case_study_path}/node_vec/protein_vec.txt'
    node_vecs = pd.read_csv(rna_vec_path, header=None).append(pd.read_csv(protein_vec_path, header=None)).reset_index(drop=True)
    dict_node_name_vec = dict(zip(node_vecs.values[:,0],node_vecs.values[:,1:]) )
    # train dataset
    # train dataset
    case_study_train_path=f'{case_study_path}/train_dataset'
    if not osp.exists(case_study_train_path):
        os.makedirs(case_study_train_path)
    else:
        shutil.rmtree(case_study_train_path,True)
        os.makedirs(case_study_train_path)
    path_pos_train =  f'{case_study_path}/case_study_train_pos_edges'
    path_neg_train =f'{case_study_path}/case_study_train_neg_edges'
    pos_train = read_interaction(path_pos_train)
    neg_train = read_interaction(path_neg_train)
    train_interactions=[]
    train_interactions.extend(pos_train)
    num_pos_train = len(train_interactions)
    train_interactions.extend(neg_train)
    num_neg_train = len(train_interactions) - num_pos_train
    y = np.ones(num_pos_train).tolist()
    y.extend(np.zeros(num_neg_train).tolist())
    train_dataset = NcRNA_Protein_Subgraph(case_study_train_path,rpin,ppin,rrsn,dict_node_name_vec,train_interactions,y,args.subgraph_type)

    # val dataset
    case_study_val_path=f'{case_study_path}/val_dataset'
    if not osp.exists(case_study_val_path):
        os.makedirs(case_study_val_path)
    else:
        shutil.rmtree(case_study_val_path,True)
        os.makedirs(case_study_val_path)
    path_pos_val =  f'{case_study_path}/case_study_val_pos_edges'
    path_neg_val =f'{case_study_path}/case_study_val_neg_edges'
    pos_val = read_interaction(path_pos_val)
    neg_val = read_interaction(path_neg_val)
    val_interactions=[]
    val_interactions.extend(pos_val)
    num_pos_val = len(val_interactions)
    val_interactions.extend(neg_val)
    num_neg_val = len(val_interactions) - num_pos_val
    y = np.ones(num_pos_val).tolist()
    y.extend(np.zeros(num_neg_val).tolist())
    val_dataset = NcRNA_Protein_Subgraph(case_study_val_path,rpin,ppin,rrsn,dict_node_name_vec,val_interactions,y,args.subgraph_type)
    print('generate case study dataset end\n')