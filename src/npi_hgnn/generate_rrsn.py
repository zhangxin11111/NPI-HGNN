import os
import numpy as np
import pandas as pd
import multiprocessing
import argparse
import os.path as osp
from src.npi_hgnn.methods import read_RPI_file
import math
import openpyxl as xl
import itertools
match = 3
mismatch = -3
gap = -2
def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
    parser.add_argument('--ratio', default=0.5,type=float)
    return parser.parse_args()
S_matrix = [[5,-4,-4,-4],[-4,5,-4,-4],[-4,-4,5,-4],[-4,-4,-4,5]]
amino_acid = ['A','U','C','G']
def read_sequence_file(path):
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
            sequence_list.append(line.strip())
    sequence_file.close()
    return name_list, sequence_list


def s_w(seqA, allseq, savepath, num):
    scorelist = [0]*(num)
    print('Comparing the %d sequence'%(num+1))
    cols = len(seqA)
    for seqB in allseq:
        rows = len(seqB)
        matrix = [[0 for row in range(rows+1)] for col in range(cols+1)]
        paths = [[0 for row in range(rows+1)] for col in range(cols+1)]
        max_score = 0
        start_pos=[1, 1]
        finalscore = 0
        for i in range(cols):
            for j in range(rows):
                a1 = amino_acid.index(seqA[i])
                a2 = amino_acid.index(seqB[j])
                s = S_matrix[a1][a2]
                if seqA[i] == seqB[j]:
                    diag = matrix[i][j] + s
                else:
                    diag = matrix[i][j] + s
                up = matrix[i + 1][j] + gap
                left = matrix[i][j + 1] + gap
                score = max(0,diag, up, left)
                matrix[i+1][j+1] = score
                if score > max_score:
                    max_score = score
                    start_pos = [i+1, j+1]
                if matrix[i+1][j+1] == diag and matrix[i+1][j+1] != 0:
                    paths[i+1][j+1] = 'diag'
                elif matrix[i+1][j+1] == up   and matrix[i+1][j+1] != 0:
                    paths[i+1][j+1] = 'up'
                elif matrix[i+1][j+1] == left and matrix[i+1][j+1] != 0:
                    paths[i+1][j+1] = 'left'
        i, j = start_pos
        start_path = paths[i][j]
        while start_path != 0:
            finalscore += matrix[i][j]
            if start_path == 'diag':
                i, j = i-1, j-1
            elif start_path == 'up':
                j = j-1
            else:
                i = i-1
            start_path = paths[i][j]
        scorelist.append(finalscore)
    np.savetxt(savepath, scorelist, delimiter=',', fmt='%f')
def generated_SW_matrix(allsequence,path):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for i in range(len(allsequence)):
        savepath = path + str(i + 1) + '.txt'
        if os.path.exists(savepath):
            continue
        sequence1 = allsequence[i]
        sequence2 = allsequence[i:]
        pool.apply_async(s_w, (sequence1, sequence2, savepath, i,))
    pool.close()
    pool.join()

    scorematrix = []
    for i in range(len(allsequence)):
        alignpath = path +str(i + 1) + '.txt'
        alignlist = pd.read_csv(alignpath, header=None, index_col=None)
        alignlist = np.array(alignlist)
        alignlist = alignlist.T
        scorematrix.append(alignlist[0])
    finalmatrix = np.array(scorematrix)
    for j in range(finalmatrix.shape[1]):
        for i in range(finalmatrix.shape[0]):
            finalmatrix[i][j] = finalmatrix[j][i]
    np.savetxt(os.path.join(path, r'rrsm.txt'), finalmatrix, delimiter=',', fmt='%f')
    return finalmatrix

if __name__ == '__main__':
    print('start generate rrsn\n')
    args = parse_args()
    rna_sequence_path=f'../../data/{args.dataset}/processed_database_data/ncRNA_sequence.fasta'
    name_list, allsequence = read_sequence_file(rna_sequence_path)
    dict_name_id = dict(zip(name_list, range(len(name_list))))
    output_path=f'../../data/{args.dataset}/source_database_data/RRSM/'
    if not osp.exists(output_path):
        os.makedirs(output_path)
    matrix=generated_SW_matrix(allsequence,output_path)
    rrsn = []
    for rna_pair in list(itertools.combinations(name_list, 2)):
            rna1_id=dict_name_id[rna_pair[0]]
            rna2_id=dict_name_id[rna_pair[1]]
            sw_score=matrix[rna1_id,rna2_id]
            score = sw_score / max(matrix[rna1_id,rna1_id], matrix[rna2_id,rna2_id])
            rrsn.append((rna_pair[0], rna_pair[1],score))
    rrsn = sorted(rrsn, key=lambda x: x[2], reverse = True)
    rpin_path=f'../../data/{args.dataset}/processed_database_data/{args.dataset}.xlsx'
    rpin_df=read_RPI_file(rpin_path)
    rpi_num=rpin_df.shape[0]
    rri_num = len(rrsn)
    extract_rri_num=math.floor(min(rpi_num,rri_num)*args.ratio)
    rrsn=rrsn[:extract_rri_num]
    rrsn_path=f'../../data/{args.dataset}/processed_database_data/{args.dataset}_RRI.xlsx'
    out_workbook = xl.Workbook()
    out_sheet = out_workbook.active
    headers = ["RNA1 names", "RNA2 names", "Labels"]
    out_sheet.append(headers)
    for rri in rrsn:
        out_sheet.append(list(rri))
    out_workbook.save(rrsn_path)
    print('generate rrsn end\n')