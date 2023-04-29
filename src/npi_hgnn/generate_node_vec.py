# coding:utf-8
import networkx as nx
from src.npi_hgnn.methods import read_all,read_rpin
import argparse
from src.npi_hgnn.methods import generate_n2v,generate_node_vec_with_fre,generate_node_vec_with_pyfeat,generate_node_vec_only_n2v
from src.npi_hgnn.methods import generate_node_vec_only_frequency,generate_node_path,generate_pre_dataset_path
import gc
def parse_args():
	parser = argparse.ArgumentParser(description="Run node2vec.")
	# NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
	parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
	# 0:kmer || n2v 1: pyfeat || n2v 2:n2v
	parser.add_argument('--nodeVecType', default=0,type=int, help='node vector type')
	# 0：RANDOM 1：FIRE 2：RelNegNPI 3、Rel&Ran
	parser.add_argument('--samplingType', default=3,type=int, help='negative sampling type')
	# 0 subgraph3 ；1 subgraph2 ；2 subgraph1
	parser.add_argument('--subgraph_type', default=1, type=int, help='type of subgraph')
	parser.add_argument('--fold', default=0,type=int, help='which fold is this')
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
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)
	return parser.parse_args()
if __name__ == '__main__':
	print('start generate node feature vector\n')
	args=parse_args()
	node_path=generate_node_path(args.dataset,args.samplingType,args.nodeVecType)
	pre_dataset_path=generate_pre_dataset_path(args.dataset,args.samplingType)
	node_names = []
	with open(f'{pre_dataset_path}/all_node_name') as f:
		lines = f.readlines()
		for line in lines:
			node_names.append(line.strip())
	node_name_vec_dict = dict(zip(node_names, range(len(node_names))))
	graph_path = f'{pre_dataset_path}/dataset_{args.fold}/pos_train_edges'
	if args.subgraph_type==2:
		G, rna_names, protein_names = read_rpin(graph_path)
	else:
		G, rna_names, protein_names = read_all(graph_path, args.dataset)
	G.add_nodes_from(node_names)
	if args.nodeVecType==0:
		path = f'{node_path}/node2vec/{args.fold}'
		generate_n2v(G, path, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
		generate_node_vec_with_fre(args.fold,args.dataset,node_path)
	elif args.nodeVecType==1:
		path = f'{node_path}/node2vec/{args.fold}'
		generate_n2v(G, path, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
		generate_node_vec_with_pyfeat(args.fold,args.dataset,node_path)
	elif args.nodeVecType == 2:
		path = f'{node_path}/node2vec/{args.fold}'
		generate_n2v(G, path, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
		generate_node_vec_only_n2v(args.fold,args.dataset,node_path)
	else:
		path = f'{node_path}/node2vec/{args.fold}'
		generate_node_vec_only_frequency(args.fold,args.dataset,node_path)
	del G
	gc.collect()
	print('generate node feature vector end\n')