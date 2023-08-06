# coding:utf-8
import argparse
import os.path as osp
import os
from src.npi_hgnn.methods import read_rpi,random_negative_sampling,write_interactor
from src.npi_hgnn.methods import fire_negative_sampling,read_sequence_file,reliable_negative_sampling
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Negative sample selection and partitioning the dataset.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN3265", help='dataset name')
    # 0：RANDOM 1：FIRE 2：RelNegNPI 3、Rel&Ran
    parser.add_argument('--samplingType',default=3,type=int, help='sampling type')
    return parser.parse_args()

def read_file(path):
    result = set()
    with open(path) as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            result.add((arr[0],arr[1]))
    return result
def write_file(path,result):
    with open(path,'w') as f:
        for line in result:
            f.write(f'{line[0]},{line[1]}\n')
def split_train_edges(path):
    case_study_pos_edges = read_file(f'{path}/case_study_pos_edges')
    case_study_pos_edges = list(case_study_pos_edges)
    case_study_train_pos_edges = case_study_pos_edges[:int(len(case_study_pos_edges)*0.9)]
    case_study_pos_edges = set(case_study_pos_edges)
    case_study_train_pos_edges = set(case_study_train_pos_edges)
    case_study_val_pos_edges = case_study_pos_edges-case_study_train_pos_edges
    write_file(f'{path}/case_study_train_pos_edges',case_study_train_pos_edges)
    write_file(f'{path}/case_study_val_pos_edges',case_study_val_pos_edges)

    case_study_neg_edges = read_file(f'{path}/case_study_neg_edges')
    case_study_neg_edges = list(case_study_neg_edges)
    case_study_train_neg_edges = case_study_neg_edges[:int(len(case_study_neg_edges)*0.9)]
    case_study_neg_edges = set(case_study_neg_edges)
    case_study_train_neg_edges = set(case_study_train_neg_edges)
    case_study_val_neg_edges = case_study_neg_edges-case_study_train_neg_edges
    write_file(f'{path}/case_study_train_neg_edges',case_study_train_neg_edges)
    write_file(f'{path}/case_study_val_neg_edges',case_study_val_neg_edges)

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
    if not osp.exists(f'../../data/{args.dataset}/case_study'):
        os.makedirs(f'../../data/{args.dataset}/case_study')
    with open(f'../../data/{args.dataset}/case_study/all_node_name',mode='w') as f:
        for item in node_name_set:
            f.write(item+'\n')
    pp_swscore_matrix = pd.read_csv(f'../../data/{args.dataset}/source_database_data/PPSM/ppsm.txt',header=None).values
    rr_swscore_matrix = pd.read_csv(f'../../data/{args.dataset}/source_database_data/RRSM/rrsm.txt',header=None).values
    set_interaction = [(triplet[0], triplet[2]) for triplet in positive_samples.values]
    set_interaction = set(set_interaction)
    if args.samplingType==0:
        set_negativeInteraction = random_negative_sampling(set_interaction, rna_name_set, protein_name_set, len(set_interaction))
    elif  args.samplingType==1:
        Positives, Negatives = fire_negative_sampling(set_interaction, rna_name_set,protein_name_set, pp_swscore_matrix,dict_protein_name_id,len(set_interaction))
        set_interaction = set(Positives)
        set_negativeInteraction = set(Negatives)
    elif  args.samplingType==2:
        Positives, Negatives = reliable_negative_sampling(set_interaction, rna_name_set,protein_name_set, pp_swscore_matrix,dict_protein_name_id,rr_swscore_matrix,dict_rna_name_id,0.5,len(set_interaction))
        set_interaction = set(Positives)
        set_negativeInteraction = set(Negatives)
    elif args.samplingType == 3:
        Positives, reliable_negatives = reliable_negative_sampling(set_interaction, rna_name_set,protein_name_set, pp_swscore_matrix,dict_protein_name_id,rr_swscore_matrix,dict_rna_name_id,0.5,len(set_interaction)//2)
        set_interaction = set(Positives)
        set_negativeInteraction = set(reliable_negatives)
        random_negatives = random_negative_sampling(set_interaction, rna_name_set, protein_name_set,len(set_interaction)//2,case_study_neg_edges=set_negativeInteraction)
        set_negativeInteraction.update(random_negatives)
    write_interactor(set_interaction, f'../../data/{args.dataset}/case_study/case_study_pos_edges')
    write_interactor(set_negativeInteraction, f'../../data/{args.dataset}/case_study/case_study_neg_edges')
    split_train_edges(f'../../data/{args.dataset}/case_study')
    print('partition dataset end\n')