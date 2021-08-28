import time

import numpy as np
import pandas as pd
import scipy as sp
import networkx as nx
import sys
import os
from collections import OrderedDict
from itertools import combinations

ppilocation = "./ppi/ppi.txt"
def read_ppi(f=ppilocation):
    
    g = nx.Graph()
    for line in open(f):
        g.add_edges_from([[i.upper() for i in line.split()]])
    
    return g


def to_conn_matrix(ppi, genes):
    
    n = len(genes)
    c=0
    conn = sp.sparse.lil_matrix((n, n))
    for i,j in combinations(genes, 2):
        c+=1
        if ppi.has_edge(i, j):
            conn[genes[i], genes[j]]=1
    
    return conn
    
        
def generank(net, fold, d):
    fold = abs(fold)
    norm_ex = fold/max(fold)
    degrees = net.sum(axis=1)
    degrees[degrees==0] = 1
    degrees = np.array(degrees)[:, 0]
    D1 = sp.sparse.csr_matrix(np.diag(1./degrees))
    A = np.eye(net.shape[0]) - d*np.dot(net.T, D1)
    b = (1-d)*norm_ex
    r = np.linalg.solve(A, b)
    return r

if __name__ == "__main__":
    
    start = time.time()   
    
    infilelocation = "./input/"
    outfilelocation  = "./output/"
    fileAllName = os.listdir(infilelocation)
    
    for name in fileAllName:
        expr_file = infilelocation + name
        name = name.replace(".csv","")
        out_file = outfilelocation
        print("processing data...")
        ppi = read_ppi()
        expr = pd.read_csv(expr_file, index_col=0, header=0)
        genes = OrderedDict([(j, i) \
                             for i, j in enumerate(expr.index)])
        conn = to_conn_matrix(ppi, genes)
        conn = conn + conn.T
        
        print("running gene rank...")
        rank = pd.DataFrame()
        for i in expr.columns:
            print("processing sample: {}".format(i))
            fold = expr[i].values
            rank_new = generank(conn, fold, 0.5)
            pro_rank = sorted(zip(rank_new, genes), reverse=True)       
            rank[i] = [i for _,i in pro_rank]
            
        rank.to_csv(out_file+name+'.txt', header=False, index=False)
        end = time.time()
        print("{}h elased".format((end - start)/3600))
        
