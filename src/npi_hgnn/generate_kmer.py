import argparse
import os
import os.path as osp
import random
def parse_args():
    parser = argparse.ArgumentParser(description="generate k-mer")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN7317", help='dataset name')
    parser.add_argument('--kRna', default=3,type=int, help='kRna of k-mer')
    parser.add_argument('--kProtein', default=2,type=int, help='kProtein of k-mer')
    return parser.parse_args()
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
def output_rna_k_mer_file(path,name_list, sequence_list):
    k_mer_file = open(path, mode='w')
    for i in range(len(name_list)):
        name = name_list[i]
        k_mer = [sequence_list[i][j:j+args.kRna] for j in range(0,len(sequence_list[i])-args.kRna+1)]
        k_mer_file.write('>' + name + '\n')
        k_mer_file.write(','.join(k_mer)+'\n')
    k_mer_file.close()
def change_protein_sequence_20_to_7(path,protein_name_list, protein_sequence_list):

    simple_protein_file = open(path, mode='w')
    for i in range(len(protein_sequence_list)):
        sequence_list = list(protein_sequence_list[i])
        for j in range(len(sequence_list)):
            if sequence_list[j] == 'A' or sequence_list[j] == 'G' or sequence_list[j] == 'V':
                sequence_list[j] = 'A'
            elif sequence_list[j] == 'I' or sequence_list[j] == 'L' or sequence_list[j] == 'F' or sequence_list[j] == 'P':
                sequence_list[j] = 'B'
            elif sequence_list[j] == 'Y' or sequence_list[j] == 'M' or sequence_list[j] == 'T' or sequence_list[j] == 'S':
                sequence_list[j] = 'C'
            elif sequence_list[j] == 'H' or sequence_list[j] == 'N' or sequence_list[j] == 'Q' or sequence_list[j] == 'W':
                sequence_list[j] = 'D'
            elif sequence_list[j] == 'R' or sequence_list[j] == 'K':
                sequence_list[j] = 'E'
            elif sequence_list[j] == 'D' or sequence_list[j] == 'E':
                sequence_list[j] = 'F'
            elif sequence_list[j] == 'C':
                sequence_list[j] = 'G'
            elif sequence_list[j] == 'X':
                temp = random.sample(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 1)[0]
                sequence_list[j] = temp
            else:
                print('protein sequence error')
                raise Exception
        protein_sequence_list[i] = ''.join(sequence_list)
        simple_protein_file.write('>' + protein_name_list[i] + '\n')
        simple_protein_file.write(protein_sequence_list[i]+'\n')
    simple_protein_file.close()
    return protein_sequence_list
def output_protein_k_mer_file(path,name_list, sequence_list):
    k_mer_file = open(path, mode='w')
    for i in range(len(name_list)):
        name = name_list[i]
        k_mer = [sequence_list[i][j:j+args.kProtein] for j in range(0,len(sequence_list[i])-args.kProtein+1)]
        k_mer_file.write('>' + name + '\n')
        k_mer_file.write(','.join(k_mer)+'\n')
    k_mer_file.close()
if __name__ == "__main__":
    print('start generate sequence k-mer\n')
    args = parse_args()
    #rna
    input_path_rna = f'../../data/{args.dataset}/processed_database_data/ncRNA_sequence.fasta'
    rna_name_list, rna_sequence_list = read_sequence_file(path=input_path_rna)
    if not osp.exists(f'../../data/{args.dataset}/k_mer/rna'):
        os.makedirs(f'../../data/{args.dataset}/k_mer/rna')
    output_rna_k_mer_file(f'../../data/{args.dataset}/k_mer/rna/result.emb',rna_name_list, rna_sequence_list )
    #protein
    input_path_protein = f'../../data/{args.dataset}/processed_database_data/protein_sequence.fasta'
    protein_name_list, protein_sequence_list = read_sequence_file(path=input_path_protein)
    if not osp.exists(f'../../data/{args.dataset}/k_mer/protein'):
        os.makedirs(f'../../data/{args.dataset}/k_mer/protein')
    protein_sequence_list=change_protein_sequence_20_to_7(f'../../data/{args.dataset}/k_mer/protein/simple_seq.txt',protein_name_list, protein_sequence_list)
    output_protein_k_mer_file(f'../../data/{args.dataset}/k_mer/protein/result.emb',protein_name_list, protein_sequence_list)
    print('generate sequence k-mer end\n')