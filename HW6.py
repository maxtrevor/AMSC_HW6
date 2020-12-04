import numpy as np
import scipy.sparse as sp
import pickle
import time

class Node:
    def __init__(self, index, nodes):
        self.num = index
        self.label = -1
        self.dist = np.inf
        self.parent = -1
        self.td = np.inf
        self.tf = np.inf
        self.neighbors = nodes

class Graph:
    def __init__(self, mat):
        self.size = mat.shape[0]
        self.nodes = [Node(i,np.nonzero(mat[i])[1]) for i in range(self.size)]
        self.time = 0
        
    def BFS(self, s):
        for node in self.nodes:
            node.parent = -1
            node.dist = np.inf
            node.label = -1
        
        self.nodes[s].label+=1
        self.nodes[s].dist=0
                       
        Q = [self.nodes[s]]
        
        while Q:
            u = Q[0]
            ns = u.neighbors
            for n in ns:
                v = self.nodes[n]
                if v.label == -1:
                    v.label+=1
                    v.dist = u.dist+1
                    v.parent = u.num
                    Q.append(v)
            u.label+=1
            Q.pop(0)
            
        comp = np.array([ j for j in range(self.size) if self.nodes[j].label==1 ])
        dists = np.array([ self.nodes[i].dist for i in comp ])
        parents = np.array([ self.nodes[i].parent for i in comp ])        
        return (comp,dists,parents)
    
    def matrix(self):
        m = np.zeros((self.size,self.size))
        for i in range(self.size):
            m[i] = np.array([ a in self.nodes[i].neighbors for a in range(self.size)]).astype(int)
        return m
    
    def DFS(self):
        comps = []
        incomplete = True
        unvisited = np.array([ i for i in range(self.size) if self.nodes[i].label==-1 ])
        visited = []
        while incomplete:
            u = self.nodes[unvisited[0]]
            self.DFS_Visit(u)
            unvisited = np.array([ i for i in range(self.size) if self.nodes[i].label==-1 ])
            visited = np.setdiff1d(np.arange(self.size),unvisited)
            for c in comps:
                visited = np.setdiff1d(visited,c)
            comps.append(visited)
            incomplete = len(unvisited)>0
        return comps
            
    
    def DFS_Visit(self, u):
        self.time+=1
        u.td=self.time
        u.label+=1
        for n in u.neighbors:
            v = self.nodes[n]
            if v.label==-1:
                v.parent=u.num
                self.DFS_Visit(v)
        u.label+=1
        self.time+=1
        u.tf=self.time

def G(n,p):
    graph = np.zeros((n,n))
    for i in np.arange(n):
        for j in np.arange(i+1,n):
            graph[i,j] = int(np.random.rand()<p)
            graph[j,i]=graph[i,j]
    return sp.csr_matrix(graph)



# do problem 1
#should take around 200 min = 3.33 hr
    

#t0 = time.time()
#n = 1000
#r = 100
#k = 40
#zs = np.linspace(0.1,4,k)    
#arr_fracS = np.zeros((k,r))
#list_smalls = [[0 for a in range(r)] for b in range(k)]
#for i,z in enumerate(zs):
#    print('z='+str(z))
#    print(time.time()-t0)
#    p = z/(n-1)
#    for j in range(r):
#        mat = G(n,p)
#        g = Graph(mat)
#        comps = g.DFS()
#        sizes = np.array([ len(c) for c in comps ])
#        big = np.max(sizes)
#        fracS = big/n
#        smalls = np.array([ s for s in sizes if s!=big ])
#        arr_fracS[i,j]=fracS
#        list_smalls[i][j] = smalls
#        
#av_fracS = np.mean(arr_fracS,axis=1)
#av_small_size = np.zeros(k)
#
#for i in range(k):
#    small_sizes = []
#    for j in range(r):
#        for n in list_smalls[i][j]:
#            small_sizes.append(n)
#    if len(small_sizes)==0:
#        av_small_size[i]=0
#    else:
#        av_small_size[i]=np.mean(small_sizes)
#    
#dat = np.array([av_fracS, av_small_size])
#pickle.dump( dat, open( "p1dat.p", "wb" ) )
   


# do problem 2
# should take about 100 min = 1.66 hr
    
#t0 = time.time()
#z = 4
#qs = np.array([10,11,12,13])
#ns = 2**qs
#k = 100
#
#av_l = np.zeros(len(ns))
#
#for i,n in enumerate(ns):
#    p = z/(n-1)
#    mat = G(n,p)
#    g = Graph(mat)
#    inds = np.random.permutation(n)[:k]
#    d = np.array([])
#    for ind in inds:
#        (comp,dists,parents)=g.BFS(ind)
#        d = np.append(d,dists)
#    av_l[i]=np.mean(d)
#    
#pickle.dump( av_l, open( "p2dat.p", "wb" ) )
#print(time.time()-t0)


n = 2**10
z = 4
p = z/(n-1)
mat = G(n,p)
g = Graph(mat)
(comp,dists,parents)=g.BFS(0)
print(dists)
