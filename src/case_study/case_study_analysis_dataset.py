import argparse
from src.npi_hgnn.methods import read_ppin,read_rrsn,read_RPI,read_PPI,read_RRI,generate_n2v,read_sequence_file,read_node2vec_file,read_kmer_fre_file,read_rpin
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
    parser.add_argument('--analysis_file', default='./all_random_case_study_edges_with_neg', help='file path')
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
def case_study_graph(projectName):
    rna_names=set()
    protein_names=set()
    G = nx.Graph()
    rpi=read_RPI(f'../../data/{projectName}/processed_database_data/{projectName}.xlsx')
    for edge in rpi:
        rna_names.add(edge[0])
        protein_names.add(edge[1])
        G.add_edge(edge[0], edge[1])
    ppi = read_PPI(f'../../data/{projectName}/processed_database_data/{projectName}_PPI.xlsx')
    for i in ppi:
        protein_names.add(i[0])
        protein_names.add(i[1])
        G.add_edge(i[0],i[1])
    rri = read_RRI(f'../../data/{projectName}/processed_database_data/{projectName}_RRI.xlsx')
    for i in rri:
        rna_names.add(i[0])
        rna_names.add(i[1])
        G.add_edge(i[0],i[1])
    G = G.to_undirected()
    return G,rna_names,protein_names
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
    # Generating Dataset
    print('start generate case study dataset\n')
    rpin, _, _ = read_rpin(f'../../data/{projectName}/case_study/case_study_train_pos_edges')
    ppin,_=read_ppin(f'../../data/{projectName}/processed_database_data/{projectName}_PPI.xlsx')
    rrsn,_=read_rrsn(f'../../data/{projectName}/processed_database_data/{projectName}_RRI.xlsx')
    rna_vec_path = f'{case_study_path}/node_vec/rna_vec.txt'
    protein_vec_path = f'{case_study_path}/node_vec/protein_vec.txt'
    node_vecs = pd.read_csv(rna_vec_path, header=None).append(pd.read_csv(protein_vec_path, header=None)).reset_index(drop=True)
    dict_node_name_vec = dict(zip(node_vecs.values[:,0],node_vecs.values[:,1:]) )

    # test dataset
    case_study_predict_path=f'{case_study_path}/predict_dataset'
    if not osp.exists(case_study_predict_path):
        os.makedirs(case_study_predict_path)
    else:
        shutil.rmtree(case_study_predict_path,True)
        os.makedirs(case_study_predict_path)
    # path_case_study_edges =  f'./case_study_edges'
    # case_study_edges = read_interaction(path_case_study_edges)
    # y = np.ones(len(case_study_edges)).tolist()
    # path_case_study_edges =  f'./random_case_study_edges'
    # case_study_edges = read_interaction(path_case_study_edges)
    # y = np.zeros(len(case_study_edges)).tolist()
    # path_case_study_edges =  f'./all_random_case_study_edges'
    # case_study_edges = read_interaction(path_case_study_edges)
    # y = np.zeros(len(case_study_edges)).tolist()
    path_case_study_edges =  args.analysis_file
    case_study_edges = read_interaction(path_case_study_edges)
    y = np.zeros(len(case_study_edges)).tolist()
    predict_dataset = NcRNA_Protein_Subgraph(case_study_predict_path,rpin,ppin,rrsn,dict_node_name_vec,case_study_edges,y,args.subgraph_type)

    print('generate case study dataset end\n')