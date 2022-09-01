from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt

class Topology(object):
    def __init__(self, config, data_dir='./data/'):
        self.topology_file = data_dir + config.topology_file
        self.shortest_paths_file = self.topology_file +'_shortest_paths'
        self.DG = nx.DiGraph()

        self.load_topology(config)
        self.calculate_paths()

    def load_topology(self,config):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':')+2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10)+2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            #print('this is our line',link,i, s, d, w, c)
            self.link_idx_to_sd[int(i)] = (int(s),int(d))
            self.link_sd_to_idx[(int(s),int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)/config.capacity_division
            self.link_weights[int(i)] = int(w)
            self.DG.add_weighted_edges_from([(int(s),int(d),int(w))])
        
        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        #print('nodes: %d, links: %d\n'%(self.num_nodes, self.num_links))

        #nx.draw_networkx(self.DG)
        #plt.show()
    def get_each_flow_oblivious_paths(self,file_path):
        counter = 0
        flows = []
        each_flow_edges = {}
        each_flow_oblivious_paths = {}
        with open(file_path) as file:
            lines = file.readlines()
            for line in lines:
                if "@" in line:
                    line  = line.strip()
                    counter+=1
                    path = line.split("@")[0]
                    path = path.rstrip()
                    path = path[1:-1]
                    path = path.split("),")
                    #print("new line",(path))
                    new_path = []
                    if "abilene" in file_path:
                        path = path[1:-1]
                    for item in path:
                        nodes = item.split(",")
                        #print("these are nodes",nodes)
                        two_nodes = []
                        for node in nodes:
                            node =node.replace(")","")
                            node =node.replace("(","")
                            node =node.replace("h","")
                            node =node.replace("s","")
                            
                            if "abilene" in file_path:
                                two_nodes.append(int(node)-1)
                            else:
                                two_nodes.append(int(node))
                        #print("the two nodes list",two_nodes)
                        new_path.append((two_nodes[0],two_nodes[1]))
                    #print("path[1:-1]",path[1:-1])
                    #print("new_path",new_path)
                    #print("start edge %s end edge of path %s"%(new_path[0],new_path[len(new_path)-1]))
                    #print("start of path ", new_path[0][0])
                    #print("end of path ",new_path[len(new_path)-1][1])
                    start = new_path[0][0]
                    end = new_path[len(new_path)-1][1]
                    flow = (start,end)
                    #print('for flow %s path is %s'%(flow,new_path))
                    try:
                        each_flow_oblivious_paths[flow].append(new_path)
                    except:
                        each_flow_oblivious_paths[flow]=[new_path]
       
        
        
        return each_flow_oblivious_paths
    def back_up_get_each_flow_oblivious_paths(self,file_path):
        counter = 0
        flows = []
        each_flow_edges = {}
        each_flow_oblivious_paths = {}
        with open(file_path) as file:
            lines = file.readlines()
            for line in lines:
                if "@" in line:
                    line  = line.strip()
                    counter+=1
                    path = line.split("@")[0]
                    path = path.rstrip()
                    path = path[1:-1]
                    path = path.split("),")
                    #print("new line",(path))
                    new_path = []
                    for item in path[1:-1]:
                        nodes = item.split(",")
                        #print("these are nodes",nodes)
                        two_nodes = []
                        for node in nodes:
                            node =node.replace(")","")
                            node =node.replace("(","")
                            node =node.replace("h","")
                            node =node.replace("s","")
                            
                            if "abilene" in file_path:
                                two_nodes.append(int(node)-1)
                            else:
                                two_nodes.append(int(node))
                        #print("the two nodes list",two_nodes)
                        new_path.append((two_nodes[0],two_nodes[1]))
                    #print("path[1:-1]",path[1:-1])
                    #print("new_path",new_path)
                    flow = (new_path[0][0],new_path[len(new_path)-1][1])
                    #print('for flow %s path is %s'%(flow,new_path))
                    try:
                        each_flow_oblivious_paths[flow].append(new_path)
                    except:
                        each_flow_oblivious_paths[flow]=[new_path]
       
        
        
        return each_flow_oblivious_paths
    def get_oblivious_paths_each_flow_edges(self,file_path,max_move,each_flow_shortest_paths):
        counter = 0
        flows = []
        each_flow_edges = {}
        each_flow_oblivious_paths = {}
        with open(file_path) as file:
            lines = file.readlines()
            for line in lines:
                if "@" in line:
                    line  = line.strip()
                    counter+=1
                    path = line.split("@")[0]
                    path = path.rstrip()
                    path = path[1:-1]
                    path = path.split("),")
                    #print("new line",(path))
                    new_path = []
                    if "abilene" in file_path:
                        path = path[1:-1]
                   
                    for item in path:
                        nodes = item.split(",")
                        two_nodes = []
                        for node in nodes:
                            node =node.replace(")","")
                            node =node.replace("(","")
                            node =node.replace("h","")
                            node =node.replace("s","")
                            if "abilene" in file_path:
                                two_nodes.append(int(node)-1)
                            else:
                                two_nodes.append(int(node))
                        #print(two_nodes)
                        new_path.append((two_nodes[0],two_nodes[1]))
                    start = new_path[0][0]
                    end = new_path[len(new_path)-1][1]
                    flow = (start,end)
                    
                    #print('for flow %s path is %s'%(flow,new_path))
                    try:
                        each_flow_oblivious_paths[flow].append(new_path)
#                         each_flow_oblivious_paths[flow]=[new_path]
                    except:
                        each_flow_oblivious_paths[flow]=[new_path]
        #print("zero",len(list(each_flow_oblivious_paths.keys())))
#         for flow ,path in each_flow_oblivious_paths.items():
#             print("flow %s oblivious path %s"%(flow,path))
        for flow,path in each_flow_shortest_paths.items():
            if flow not in each_flow_oblivious_paths:
                print("flow %s doest have an oblivous path"%(flow))
        for flow,paths in each_flow_oblivious_paths.items():
            if len(paths)==0:
                print(flow,paths)
        each_flow_chosen_paths = {}
        chosen_paths = []
        there_is_at_least_one_path = True
        one_path_for_each_flow_flag = {}
        for flow in each_flow_shortest_paths:
            one_path_for_each_flow_flag[flow] = True
        while(len(chosen_paths)<max_move and there_is_at_least_one_path):
            there_is_at_least_one_path = False
            for flow in each_flow_shortest_paths:
                paths = each_flow_oblivious_paths[flow]
                if paths:
                    there_is_at_least_one_path = True
                    #print(paths)
                    path = paths[0]
                    if path not in chosen_paths:
                        chosen_paths.append(path)
                    paths.pop(0)
                    each_flow_oblivious_paths[flow] = paths
                    if len(chosen_paths)<max_move or one_path_for_each_flow_flag[flow]:
                        try:
                            each_flow_chosen_paths[flow].append(path)
                        except:
                            each_flow_chosen_paths[flow]=[path]
                        one_path_for_each_flow_flag[flow] = False
        
       
        
        
    
        each_flow_edges = {}
        flow_counter = 1
        for flow,paths in each_flow_chosen_paths.items():
            #print("for %s flow number %s we"%(flow,flow_counter))
            
            for path in paths:
                #print("is it normal? a list of edges??",path)
                #if len(path)>1:
                for edge in path:
                    try:
                        if (edge) not in each_flow_edges[flow]:
                            each_flow_edges[flow].append(edge)
                            each_flow_edges[flow].append((edge[1],edge[0]))
                            #print("we added for ",flow,flow_counter,edge,(edge[1],edge[0]))
                    except:
                        each_flow_edges[flow]=[edge]
                        each_flow_edges[flow].append((edge[1],edge[0]))
                            #print("we added for ",flow,flow_counter)
#                 else:
#                     one_hop_path = path[0]
#                     print("error in this path",path,one_hop_path,one_hop_path[0],one_hop_path[1])
#                     try:
                        
#                         if (one_hop_path[0],one_hop_path[1]) not in each_flow_edges[flow]:
#                             each_flow_edges[flow].append((one_hop_path[0],one_hop_path[1]))
#                             each_flow_edges[flow].append((one_hop_path[1],one_hop_path[0]))
#                             #print("we added for ",flow,flow_counter)
#                     except:

#                         each_flow_edges[flow]=[(one_hop_path[0],one_hop_path[1])]
#                         each_flow_edges[flow].append((one_hop_path[1],one_hop_path[0]))
                        #print("we added for ",flow,flow_counter)
        flow_counter+=1
        #print("third",len(list(each_flow_edges.keys())))
        return each_flow_edges
    
    def back_up_get_oblivious_paths_each_flow_edges(self,file_path,max_move,each_flow_shortest_paths):
        counter = 0
        flows = []
        each_flow_edges = {}
        each_flow_oblivious_paths = {}
        with open(file_path) as file:
            lines = file.readlines()
            for line in lines:
                if "@" in line:
                    line  = line.strip()
                    counter+=1
                    path = line.split("@")[0]
                    path = path.rstrip()
                    path = path[1:-1]
                    path = path.split("),")
                    #print("new line",(path))
                    new_path = []
                    for item in path[1:-1]:
                        nodes = item.split(",")
                        two_nodes = []
                        for node in nodes:
                            node =node.replace(")","")
                            node =node.replace("(","")
                            node =node.replace("h","")
                            node =node.replace("s","")
                            if "abilene" in file_path:
                                two_nodes.append(int(node)-1)
                            else:
                                two_nodes.append(int(node))
                        #print(two_nodes)
                        new_path.append((two_nodes[0],two_nodes[1]))

                    flow = (new_path[0][0],new_path[len(new_path)-1][1])
                    #print('for flow %s path is %s'%(flow,new_path))
                    try:
                        each_flow_oblivious_paths[flow].append(new_path)
#                         each_flow_oblivious_paths[flow]=[new_path]
                    except:
                        each_flow_oblivious_paths[flow]=[new_path]
        #print("zero",len(list(each_flow_oblivious_paths.keys())))
#         for flow ,path in each_flow_oblivious_paths.items():
#             print("flow %s oblivious path %s"%(flow,path))
        for flow,path in each_flow_shortest_paths.items():
            if flow not in each_flow_oblivious_paths:
                print("flow %s doest have an oblivous path"%(flow))
        for flow,paths in each_flow_oblivious_paths.items():
            if len(paths)==0:
                print(flow,paths)
        each_flow_chosen_paths = {}
        chosen_paths = []
        there_is_at_least_one_path = True
        while(len(chosen_paths)<max_move and there_is_at_least_one_path):
            there_is_at_least_one_path = False
            for flow in each_flow_shortest_paths:
                paths = each_flow_oblivious_paths[flow]
                if paths:
                    there_is_at_least_one_path = True
                    #print(paths)
                    path = paths[0]
                    if path not in chosen_paths:
                        chosen_paths.append(path)
                    paths.pop(0)
                    each_flow_oblivious_paths[flow] = paths
                    try:
                        each_flow_chosen_paths[flow].append(path)
                    except:
                        each_flow_chosen_paths[flow]=[path]
        
       
        
        
    
        each_flow_edges = {}
        flow_counter = 1
        for flow,paths in each_flow_chosen_paths.items():
            #print("for %s flow number %s we"%(flow,flow_counter))
            
            for path in paths:
                #print("is it normal? a list of edges??",path)
                #if len(path)>1:
                for edge in path:
                    try:
                        if (edge) not in each_flow_edges[flow]:
                            each_flow_edges[flow].append(edge)
                            each_flow_edges[flow].append((edge[1],edge[0]))
                            #print("we added for ",flow,flow_counter,edge,(edge[1],edge[0]))
                    except:
                        each_flow_edges[flow]=[edge]
                        each_flow_edges[flow].append((edge[1],edge[0]))
                            #print("we added for ",flow,flow_counter)
#                 else:
#                     one_hop_path = path[0]
#                     print("error in this path",path,one_hop_path,one_hop_path[0],one_hop_path[1])
#                     try:
                        
#                         if (one_hop_path[0],one_hop_path[1]) not in each_flow_edges[flow]:
#                             each_flow_edges[flow].append((one_hop_path[0],one_hop_path[1]))
#                             each_flow_edges[flow].append((one_hop_path[1],one_hop_path[0]))
#                             #print("we added for ",flow,flow_counter)
#                     except:

#                         each_flow_edges[flow]=[(one_hop_path[0],one_hop_path[1])]
#                         each_flow_edges[flow].append((one_hop_path[1],one_hop_path[0]))
                        #print("we added for ",flow,flow_counter)
        flow_counter+=1
        #print("third",len(list(each_flow_edges.keys())))
        return each_flow_edges
       
    def get_each_flow_shortest_paths(self):
        each_flow_shortest_path ={}
        if os.path.exists(self.shortest_paths_file):
            #print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>')+1:])
                self.pair_idx_to_sd.append((s,d))
                self.pair_sd_to_idx[(s,d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':')+1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx+3:]
                    #print("for flow %s -> %s we have paths %s type %s"%(s,d,node_path,type(node_path)))
                    each_flow_shortest_path[(s,d)] = list(node_path)
        return each_flow_shortest_path      
    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>')+1:])
                self.pair_idx_to_sd.append((s,d))
                self.pair_sd_to_idx[(s,d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':')+1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx+3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s,d))
                        self.pair_sd_to_idx[(s,d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.DG, s, d, weight='weight')))
                        line = str(s)+'->'+str(d)+': '+str(self.shortest_paths[-1])
                        f.writelines(line+'\n')
        #print("this could be a check point pairs %s nodeds %s"%(self.num_pairs,self.num_nodes))
        assert self.num_pairs == self.num_nodes*(self.num_nodes-1)
        f.close()
        
        print('pairs: %d, nodes: %d, links: %d\n'\
                %(self.num_pairs, self.num_nodes, self.num_links))

  

class Traffic(object):
    def __init__(self, config, num_nodes, data_dir='./data/', is_training=False):
        if is_training:
            self.traffic_file = data_dir + config.topology_file + config.traffic_file
        else:
            self.traffic_file = data_dir + config.topology_file + config.test_traffic_file
        self.num_nodes = num_nodes
        self.load_traffic(config)

    def load_traffic(self, config):
        assert os.path.exists(self.traffic_file)
        print('[*] Loading traffic matrices...', self.traffic_file)
        f = open(self.traffic_file, 'r')
        traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            assert total_volume_cnt == self.num_nodes*self.num_nodes
            matrix = np.zeros((self.num_nodes, self.num_nodes))
            for v in range(total_volume_cnt):
                i = int(v/self.num_nodes)
                j = v%self.num_nodes
                if i != j:
                    #print('from node %s to node %s we have %s trafffic'%(i,j,float(volumes[v])*10))
                    matrix[i][j] = float(volumes[v])*config.demand_scale
            #print(matrix + '\n')
            traffic_matrices.append(matrix)

        f.close()
        self.traffic_matrices = np.array(traffic_matrices)

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        print('Traffic matrices dims: [%d, %d, %d]\n'%(tms_shape[0], tms_shape[1], tms_shape[2]))


    
class Environment(object):
    def __init__(self, config, is_training=False):
        self.data_dir = './data/'
        self.topology = Topology(config, self.data_dir)
        self.traffic = Traffic(config, self.topology.num_nodes, self.data_dir, is_training=is_training)
        #print("this is the actual traffic",self.traffic.traffic_matrices)
        self.traffic_matrices = self.traffic.traffic_matrices*100*8/300/1000    #kbps
        #print("this is kbps",self.traffic_matrices)
        import pdb
        #pdb.set_trace()
        self.tm_cnt = self.traffic.tm_cnt
        self.traffic_file = self.traffic.traffic_file
        self.num_pairs = self.topology.num_pairs
        self.pair_idx_to_sd = self.topology.pair_idx_to_sd
        self.pair_sd_to_idx = self.topology.pair_sd_to_idx
        self.num_nodes = self.topology.num_nodes
        self.num_links = self.topology.num_links
        self.link_idx_to_sd = self.topology.link_idx_to_sd
        self.link_sd_to_idx = self.topology.link_sd_to_idx
        self.link_capacities = self.topology.link_capacities
        self.link_weights = self.topology.link_weights
        self.shortest_paths_node = self.topology.shortest_paths                         # paths consist of nodes
        self.shortest_paths_link = self.convert_to_edge_path(self.shortest_paths_node)  # paths consist of links

    def convert_to_edge_path(self, node_paths):
        edge_paths = []
        num_pairs = len(node_paths)
        for i in range(num_pairs):
            edge_paths.append([])
            num_paths = len(node_paths[i])
            for j in range(num_paths):
                edge_paths[i].append([])
                path_len = len(node_paths[i][j])
                for n in range(path_len-1):
                    e = self.link_sd_to_idx[(node_paths[i][j][n], node_paths[i][j][n+1])]
                    assert e>=0 and e<self.num_links
                    edge_paths[i][j].append(e)
                #print(i, j, edge_paths[i][j])

        return edge_paths
