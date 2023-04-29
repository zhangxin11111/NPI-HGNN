import os
import numpy as np
import pandas as pd
import multiprocessing
import argparse
import os.path as osp
match = 3
mismatch = -3
gap = -2
def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
    return parser.parse_args()

S_matrix = [[9,-1,-1,-3,0,-3,-3,-3,-4,-3,-3,-3,-3,-1,-1,-1,-1,-2,-2,-2],
     [-1,4,1,-1,1,0,1,0,0,0,-1,-1,0,-1,-2,-2,-2,-2,-2,-3],
     [-1,1,4,1,-1,1,0,1,0,0,0,-1,0,-1,-2,-2,-2,-2,-2,-3],
     [-3,-1,1,7,-1,-2,-1,-1,-1,-1,-2,-2,-1,-2,-3,-3,-2,-4,-3,-4],
     [0,1,-1,-1,4,0,-1,-2,-1,-1,-2,-1,-1,-1,-1,-1,-2,-2,-2,-3],
     [-3,0,1,-2,0,6,-2,-1,-2,-2,-2,-2,-2,-3,-4,-4,0,-3,-3,-2],
     [-3,1,0,-2,-2,0,6,1,0,0,-1,0,0,-2,-3,-3,-3,-3,-2,-4],
     [-3,0,1,-1,-2,-1,1,6,2,0,-1,-2,-1,-3,-3,-4,-3,-3,-3,-4],
     [-4,0,0,-1,-1,-2,0,2,5,2,0,0,1,-2,-3,-3,-3,-3,-2,-3],
     [-3,0,0,-1,-1,-2,0,0,2,5,0,1,1,0,-3,-2,-2,-3,-1,-2],
     [-3,-1,0,-2,-2,-2,1,1,0,0,8,0,-1,-2,-3,-3,-2,-1,2,-2],
     [-3,-1,-1,-2,-1,-2,0,-2,0,1,0,5,2,-1,-3,-2,-3,-3,-2,-3],
     [-3,0,0,-1,-1,-2,0,-1,1,1,-1,2,5,-1,-3,-2,-3,-3,-2,-3],
     [-1,-1,-1,-2,-1,-3,-2,-3,-2,0,-2,-1,-1,5,1,2,-2,0,-1,-1],
     [-1,-2,-2,-3,-1,-4,-3,-3,-3,-3,-3,-3,-3,1,4,2,1,0,-1,-3],
     [-1,-2,-2,-3,-1,-4,-3,-4,-3,-2,-3,-2,-2,2,2,4,3,0,-1,-2],
     [-1,-2,-2,-2,0,-3,-3,-3,-2,-2,-3,-3,-2,1,3,1,4,-1,-1,-3],
     [-2,-2,-2,-4,-2,-3,-3,-3,-3,-3,-1,-3,-3,0,0,0,-1,6,3,1],
     [-2,-2,-2,-3,-2,-3,-2,-3,-2,-1,2,-2,-2,-1,-1,-1,-1,3,7,2],
     [-2,-3,-3,-4,-3,-2,-4,-4,-3,-2,-2,-3,-3,-1,-3,-2,-3,1,2,11]]

amino_acid = ['C','S','T','P','A', 'G', 'N','D','E','Q','H','R','K',
            'M','I','L','V','F','Y','W']
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
        start_pos = [1, 1]
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

def generated_SW_matrix(filename,path):
    name_list, allsequence = read_sequence_file(path=filename)


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
        alignpath = path + str(i + 1) + '.txt'
        alignlist = pd.read_csv(alignpath, header=None, index_col=None)
        alignlist = np.array(alignlist)
        alignlist = alignlist.T
        scorematrix.append(alignlist[0])
    finalmatrix = np.array(scorematrix)
    for j in range(finalmatrix.shape[1]):
        for i in range(finalmatrix.shape[0]):
            finalmatrix[i][j] = finalmatrix[j][i]
    np.savetxt(os.path.join(path, r'ppsm.txt'), finalmatrix, delimiter=',', fmt='%f')

if __name__ == '__main__':
    print('start generate ppsm\n')
    args = parse_args()
    protein_sequence_path=f'../../data/{args.dataset}/processed_database_data/protein_sequence.fasta'
    output_path=f'../../data/{args.dataset}/source_database_data/PPSM/'
    if not osp.exists(output_path):
        os.makedirs(output_path)
    generated_SW_matrix(protein_sequence_path,output_path)
    print('generate ppsm end\n')