import itertools
import os
import os.path as osp
import argparse
from sklearn.decomposition import PCA
import numpy as np
import joblib
import random
DNAelements = 'ACGT'
RNAelements = 'ACGU'
proteinElements = 'ABCDEFG'#'ACDEFGHIKLMNPQRSTVWY'
def parse_args():
    p = argparse.ArgumentParser(description='Features Geneation Tool from DNA, RNA, and Protein Sequences')
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    p.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
    p.add_argument('-kgap', '--kGap', type=int, help='(l,k,p)-mers', default=5)
    p.add_argument('-ktuple', '--kTuple', type=int, help='k=1 then (X), k=2 then (XX), k=3 then (XXX),', default=3)
    p.add_argument('-full', '--fullDataset', type=int, help='saved full dataset', default=0, choices=[0, 1])
    p.add_argument('-test', '--testDataset', type=int, help='saved test dataset', default=0, choices=[0, 1])
    p.add_argument('-optimum', '--optimumDataset', type=int, help='saved optimum dataset', default=1, choices=[0, 1])
    p.add_argument('-pseudo', '--pseudoKNC', type=int, help='Generate feature: X, XX, XXX, XXX', default=1, choices=[0, 1])
    p.add_argument('-zcurve', '--zCurve', type=int, help='x_, y_, z_', default=1, choices=[0, 1])
    p.add_argument('-gc', '--gcContent', type=int, help='GC/ACGT', default=1, choices=[0, 1])
    p.add_argument('-skew', '--cumulativeSkew', type=int, help='GC, AT', default=1, choices=[0, 1])
    p.add_argument('-atgc', '--atgcRatio', type=int, help='atgcRatio', default=1, choices=[0, 1])
    p.add_argument('-f11', '--monoMono', type=int, help='Generate feature: X_X', default=1, choices=[0, 1])
    p.add_argument('-f12', '--monoDi', type=int, help='Generate feature: X_XX', default=1, choices=[0, 1])
    p.add_argument('-f13', '--monoTri', type=int, help='Generate feature: X_XXX', default=1, choices=[0, 1])
    p.add_argument('-f21', '--diMono', type=int, help='Generate feature: XX_X', default=1, choices=[0, 1])
    p.add_argument('-f22', '--diDi', type=int, help='Generate feature: XX_XX', default=1, choices=[0, 1])
    p.add_argument('-f23', '--diTri', type=int, help='Generate feature: XX_XXX', default=1, choices=[0, 1])
    p.add_argument('-f31', '--triMono', type=int, help='Generate feature: XXX_X', default=1, choices=[0, 1])
    p.add_argument('-f32', '--triDi', type=int, help='Generate feature: XXX_XX', default=1, choices=[0, 1])
    return p.parse_args()
def sequenceType(seqType):
    if seqType == 'DNA':
        elements = DNAelements
    else:
        if seqType == 'RNA':
            elements = RNAelements
        else:
            if seqType == 'PROTEIN':
                elements = proteinElements
            else:
                elements = None
    return elements
def gF(args, sequence_list):
    elements = sequenceType(args.sequenceType.upper())
    m2 = list(itertools.product(elements, repeat=2))
    m3 = list(itertools.product(elements, repeat=3))
    m4 = list(itertools.product(elements, repeat=4))
    m5 = list(itertools.product(elements, repeat=5))
    T = []  # All instance ...
    def kmers(seq, k):
        v = []
        for i in range(len(seq) - k + 1):
            v.append(seq[i:i + k])
        return v
    def pseudoKNC(x, k):
        for i in range(1, k + 1, 1):
            v = list(itertools.product(elements, repeat=i))
            for i in v:
                t.append(x.count(''.join(i)))
    def zCurve(x, seqType):
        if seqType == 'DNA' or seqType == 'RNA':
            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None
            A = x.count('A'); C = x.count('C'); G = x.count('G');
            x_ = (A + G) - (C + TU)
            y_ = (A + C) - (G + TU)
            z_ = (A + TU) - (C + G)
            t.append(x_); t.append(y_); t.append(z_)
    def gcContent(x, seqType):
        if seqType == 'DNA' or seqType == 'RNA':
            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None
            A = x.count('A');
            C = x.count('C');
            G = x.count('G');
            t.append( f'{(G + C) / (A + C + G + TU)  * 100.0:.2f}' )
    def cumulativeSkew(x, seqType):
        if seqType == 'DNA' or seqType == 'RNA':
            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None
            A = x.count('A');
            C = x.count('C');
            G = x.count('G');
            if G+C==0:
                GCSkew=G-C
            else:
                GCSkew = (G-C)/(G+C)
            if A + TU==0:
                ATSkew = (A - TU)
            else:
                ATSkew = (A - TU) / (A + TU)
            t.append(f'{GCSkew:.2f}')
            t.append(f'{ATSkew:.2f}')
    def atgcRatio(x, seqType):
        if seqType == 'DNA' or seqType == 'RNA':
            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None
            A = x.count('A');
            C = x.count('C');
            G = x.count('G');
            if G+C==0:
                ratio=(A+TU)
            else:
                ratio = (A+TU)/(G+C)
            t.append(f'{ratio:.2f}')
    def monoMonoKGap(x, g):  # 1___1
        m = m2
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 2)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[-1] == gGap[1]:
                        C += 1
                t.append(C)
    def monoDiKGap(x, g):  # 1___2
        m = m3
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 3)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                        C += 1
                t.append(C)
    def diMonoKGap(x, g):  # 2___1
        m = m3
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 3)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[1] == gGap[1] and v[-1] == gGap[2]:
                        C += 1
                t.append(C)
    def monoTriKGap(x, g):  # 1___3
        m = m4
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 4)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[-3] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                        C += 1
                t.append(C)
    def triMonoKGap(x, g):  # 3___1
        m = m4
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 4)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-1] == gGap[3]:
                        C += 1
                t.append(C)
    def diDiKGap(x, g):
        m = m4
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 4)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[1] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                        C += 1
                t.append(C)
    def diTriKGap(x, g):  # 2___3
        m = m5
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 5)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[1] == gGap[1] and v[-3] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                        C += 1
                t.append(C)
    def triDiKGap(x, g):  # 3___2
        m = m5
        for i in range(1, g + 1, 1):
            V = kmers(x, i + 5)
            for gGap in m:
                C = 0
                for v in V:
                    if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                        C += 1
                t.append(C)
    def generateFeatures(kGap, kTuple, x):
        if args.zCurve == 1:
            zCurve(x, args.sequenceType.upper())              #3
        if args.gcContent == 1:
            gcContent(x, args.sequenceType.upper())           #1
        if args.cumulativeSkew == 1:
            cumulativeSkew(x, args.sequenceType.upper())      #2
        if args.atgcRatio == 1:
            atgcRatio(x, args.sequenceType.upper())         #1
        if args.pseudoKNC == 1:
            pseudoKNC(x, kTuple)            #k=2|(16), k=3|(64), k=4|(256), k=5|(1024);
        if args.monoMono == 1:
            monoMonoKGap(x, kGap)      #4*(k)*4 = 240
        if args.monoDi == 1:
            monoDiKGap(x, kGap)        #4*k*(4^2) = 960
        if args.monoTri == 1:
            monoTriKGap(x, kGap)       #4*k*(4^3) = 3,840
        if args.diMono == 1:
            diMonoKGap(x, kGap)        #(4^2)*k*(4)    = 960
        if args.diDi == 1:
            diDiKGap(x, kGap)          #(4^2)*k*(4^2)  = 3,840
        if args.diTri == 1:
            diTriKGap(x, kGap)         #(4^2)*k*(4^3)  = 15,360
        if args.triMono == 1:
            triMonoKGap(x, kGap)       #(4^3)*k*(4)    = 3,840
        if args.triDi == 1:
            triDiKGap(x, kGap)         #(4^3)*k*(4^2)  = 15,360
    for x in sequence_list:
        t = []
        generateFeatures(args.kGap, args.kTuple, x)
        T.append(t)
    return np.array(T)
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
def change_protein_sequence_20_to_7(protein_sequence_list):

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
    return protein_sequence_list
if __name__ == "__main__":
    print('start sequence generate pyfeat\n')
    args = parse_args()
    sequence_file_path = f'../../data/{args.dataset}/processed_database_data/ncRNA_sequence.fasta'
    name_list, sequence_list=read_sequence_file(sequence_file_path)
    args.sequenceType = 'RNA' # DNA/RNA/PROTEIN
    T = gF(args, sequence_list)
    pca = PCA(n_components=100)
    rna_vec=np.array(T)
    if not osp.exists(f'../../data/{args.dataset}/pyfeat/rna_seq'):
        os.makedirs(f'../../data/{args.dataset}/pyfeat/rna_seq')
    if not osp.exists(f'../../data/{args.dataset}/pyfeat/rna_seq/model'):
        rna_vec = pca.fit_transform(rna_vec)
        joblib.dump(pca, f'../../data/{args.dataset}/pyfeat/rna_seq/model')
    else:
        pca_model = joblib.load(f'../../data/{args.dataset}/pyfeat/rna_seq/model')
        rna_vec = pca_model.transform(rna_vec)
    print(rna_vec.shape)
    with open(f'../../data/{args.dataset}/pyfeat/rna_seq/result.emb','w') as f:
        for x, y in zip(rna_vec.tolist(), name_list):
            f.write(y+',')
            f.write(','.join(list(map(str,x)))+'\n')
    sequence_file_path = f'../../data/{args.dataset}/processed_database_data/protein_sequence.fasta'
    name_list, sequence_list = read_sequence_file(sequence_file_path)
    sequence_list = change_protein_sequence_20_to_7(sequence_list)
    args.sequenceType = 'PROTEIN'  # DNA/RNA/PROTEIN
    T = gF(args, sequence_list)
    pca = PCA(n_components=100)
    rna_vec = np.array(T)
    if not osp.exists(f'../../data/{args.dataset}/pyfeat/protein_seq'):
        os.makedirs(f'../../data/{args.dataset}/pyfeat/protein_seq')
    if not osp.exists(f'../../data/{args.dataset}/pyfeat/protein_seq/model'):
        rna_vec = pca.fit_transform(rna_vec)
        joblib.dump(pca, f'../../data/{args.dataset}/pyfeat/protein_seq/model')
    else:
        pca_model = joblib.load(f'../../data/{args.dataset}/pyfeat/protein_seq/model')
        rna_vec = pca_model.transform(rna_vec)
    print(rna_vec.shape)
    with open(f'../../data/{args.dataset}/pyfeat/protein_seq/result.emb', 'w') as f:
        for x, y in zip(rna_vec.tolist(), name_list):
            f.write(y + ',')
            f.write(','.join(list(map(str, x))) + '\n')
    print('generate sequence pyfeat end\n')