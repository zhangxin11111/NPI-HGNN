import argparse
import os
import os.path as osp
from gensim.models import Word2Vec
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="generate word2vec")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN7317", help='dataset name')
    return parser.parse_args()
def read_k_mer_file(path):

    name_list = []
    sequence_list = []
    sequence_file_path = path
    sequence_file = open(sequence_file_path, mode='r')
    for line in sequence_file.readlines():
        if line[0].strip()=='':
            continue
        if line[0] == '>':
            sequence_name = line.strip()[1:]
            name_list.append(sequence_name)
        else:
            sequence_list.append(line.strip().split(','))
    sequence_file.close()
    return name_list, sequence_list
def generate_rna_k_mer_vec():

    name_list, sequence_list = read_k_mer_file(f'../../data/{args.dataset}/k_mer/rna/result.emb')
    if not osp.exists(f'../../data/{args.dataset}/word2vec/rna_kmer'):
        os.makedirs(f'../../data/{args.dataset}/word2vec/rna_kmer')
    model = Word2Vec(sequence_list,min_count=1)
    model.wv.save_word2vec_format(f'../../data/{args.dataset}/word2vec/rna_kmer/result.emb')
def read_k_mer_vec_file(path):

    k_mers = []
    k_mer_vecs = []
    k_mer_vec_path = path
    k_mer_vec_file = open(k_mer_vec_path, mode='r')
    lines = k_mer_vec_file.readlines()
    lines.pop(0)
    for line in lines:
        if line[0].strip()=='':
            continue
        temp=line.strip().split(' ')
        k_mers.append(temp[0])
        k_mer_vecs.append(temp[1:])
    k_mer_vec_file.close()
    return k_mers, k_mer_vecs
def get_rna_k_mer_fre():
    k_mers, k_mer_vecs=read_k_mer_vec_file(f'../../data/{args.dataset}/word2vec/rna_kmer/result.emb')
    name_list, sequence_list = read_k_mer_file(f'../../data/{args.dataset}/k_mer/rna/result.emb')
    if not osp.exists(f'../../data/{args.dataset}/word2vec/rna_seq'):
        os.makedirs(f'../../data/{args.dataset}/word2vec/rna_seq')
    rna_seq_file = open(f'../../data/{args.dataset}/word2vec/rna_seq/result.emb', mode='w')
    if not osp.exists(f'../../data/{args.dataset}/frequency/rna_seq'):
        os.makedirs(f'../../data/{args.dataset}/frequency/rna_seq')
    rna_fre_file=open(f'../../data/{args.dataset}/frequency/rna_seq/result.emb', mode='w')
    for i in range(len(name_list)):
        my_dict = {value: 0 for value in k_mers}
        name = name_list[i]
        seq_k_mers=sequence_list[i]
        for k_mer in seq_k_mers:
            my_dict[k_mer]+=1
        rna_seq_vec=np.array([])
        for j,value in enumerate(k_mers):
            k_mer_vec=[float(m)*my_dict[value] for m in k_mer_vecs[j]]
            rna_seq_vec=np.append(rna_seq_vec,k_mer_vec)
        rna_seq_vec=rna_seq_vec.astype(np.str_)
        rna_seq_file.write( name+',' +','.join(rna_seq_vec.tolist()) + '\n')
        rna_fre_file.write(name+','+','.join([str(y) for x,y in my_dict.items()])+'\n')
    rna_seq_file.close()
    rna_fre_file.close()
def generate_protein_k_mer_vec():

    name_list, sequence_list = read_k_mer_file(f'../../data/{args.dataset}/k_mer/protein/result.emb')
    if not osp.exists(f'../../data/{args.dataset}/word2vec/protein_kmer'):
        os.makedirs(f'../../data/{args.dataset}/word2vec/protein_kmer')
    model = Word2Vec(sequence_list,min_count=1)
    model.wv.save_word2vec_format(f'../../data/{args.dataset}/word2vec/protein_kmer/result.emb')
def get_protein_k_mer_fre():
    k_mers, k_mer_vecs=read_k_mer_vec_file(f'../../data/{args.dataset}/word2vec/protein_kmer/result.emb')
    name_list, sequence_list = read_k_mer_file(f'../../data/{args.dataset}/k_mer/protein/result.emb')
    if not osp.exists(f'../../data/{args.dataset}/word2vec/protein_seq'):
        os.makedirs(f'../../data/{args.dataset}/word2vec/protein_seq')
    protein_seq_file = open(f'../../data/{args.dataset}/word2vec/protein_seq/result.emb', mode='w')
    if not osp.exists(f'../../data/{args.dataset}/frequency/protein_seq'):
        os.makedirs(f'../../data/{args.dataset}/frequency/protein_seq')
    protein_fre_file=open(f'../../data/{args.dataset}/frequency/protein_seq/result.emb', mode='w')
    for i in range(len(name_list)):
        my_dict = {value: 0 for value in k_mers}
        name = name_list[i]
        seq_k_mers=sequence_list[i]
        for k_mer in seq_k_mers:
            my_dict[k_mer]+=1
        protein_seq_vec=np.array([])
        for j,value in enumerate(k_mers):
            k_mer_vec=[float(m)*my_dict[value] for m in k_mer_vecs[j]]
            protein_seq_vec=np.append(protein_seq_vec,k_mer_vec)
        protein_seq_vec=protein_seq_vec.astype(np.str_)
        protein_seq_file.write( name+',' +','.join(protein_seq_vec.tolist()) + '\n')
        protein_fre_file.write(name+','+','.join([str(y) for x,y in my_dict.items()])+'\n')
    protein_seq_file.close()
    protein_fre_file.close()
if __name__ == "__main__":
    print('start generate sequence frequency feature\n')
    args = parse_args()
    generate_rna_k_mer_vec()
    get_rna_k_mer_fre()
    generate_protein_k_mer_vec()
    get_protein_k_mer_fre()
    print('generate sequence frequency feature end\n')