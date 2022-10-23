#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import csv
from tqdm import tqdm
import numpy as np
import time
from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value, GLPK

OBJ_EPSILON = 1e-12


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


class Game(object):
    def __init__(self, config, env, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)
 
        self.data_dir = env.data_dir
        self.DG = env.topology.DG
        self.traffic_file = env.traffic_file
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.num_pairs = env.num_pairs
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.shortest_paths_node = env.shortest_paths_node              # paths with node info
        self.shortest_paths_link = env.shortest_paths_link              # paths with link info

        self.get_ecmp_next_hops()
        
        self.model_type = config.model_type
        
        #for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]

        self.load_multiplier = {}
        
        
        self.each_path_edges = {}
        self.each_flow_paths = {}
        self.each_flow_shortest_path ={}
        self.each_path_id={}
        self.each_id_path={}
        
        
    def generate_inputs(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros((self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], self.tm_history), dtype=np.float32)   #tm state  [Valid_tms, Node, Node, History]
        idx_offset = self.tm_history - 1
        for tm_idx in self.tm_indexes:
            for h in range(self.tm_history):
                if normalization:
                    tm_max_element = np.max(self.traffic_matrices[tm_idx-h])
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h] / tm_max_element        #[Valid_tms, Node, Node, History]
                else:
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h]                         #[Valid_tms, Node, Node, History]

    def get_topK_flows(self, tm_idx, pairs):
        tm = self.traffic_matrices[tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]

        sorted_f = sorted(f.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)

        cf = []
        for i in range(self.max_moves):
            cf.append(sorted_f[i][0])

        return cf
       
    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    def ecmp_next_hop_distribution(self, link_loads, demand, src, dst):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)
        #if next_hops_cnt > 1:
            #print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])

        ecmp_demand = demand / next_hops_cnt 
        for np in ecmp_next_hops:
            link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
            self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst)

    def ecmp_traffic_distribution(self, tm_idx):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, d)

        return link_loads

    def get_critical_topK_flows(self, tm_idx, critical_links=5):
        link_loads = self.ecmp_traffic_distribution(tm_idx)
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:critical_links]
        
        cf_potential = []
        for pair_idx in range(self.num_pairs):
            for path in self.shortest_paths_link[pair_idx]:
                if len(set(path).intersection(critical_link_indexes)) > 0:
                    cf_potential.append(pair_idx)
                    break

        #print(cf_potential)
        assert len(cf_potential) >= self.max_moves,                 ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)"%(cf_potential, self.max_moves, critical_links))

        return self.get_topK_flows(tm_idx, cf_potential)
    def eval_ecmp_traffic_distribution2(self, tm_idx, look_ahead_window,eval_delay=False):
        mlus = []
        for tm in range(tm_idx,tm_idx+look_ahead_window):
            eval_link_loads = self.ecmp_traffic_distribution(tm)
            eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
            self.load_multiplier[tm] = 0.9 / eval_max_utilization
            mlus.append(eval_max_utilization)
        
            
        return mlus        
    def eval_ecmp_traffic_distribution(self, tm_idx, eval_delay=False):
        eval_link_loads = self.ecmp_traffic_distribution(tm_idx)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay
    def optimal_routing_mlu_default(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]

        demands = {}
        

        model = LpProblem(name="routing")
        #print('this %s is our self.pair_links' %(self.pair_links)) 
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
        #print(ratio)
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)
        #print("this %s is self.lp_links"%(self.lp_links))
        #print("this %s is self.lp_nodes"%(self.lp_nodes))
        #print("this %s is self.lp_pairs"%(self.lp_pairs))
        """this will give MLU 0.5"""
        each_flow_edges = {0:[(0,1)],1:[(0,2)],2:[(0,1),(1,3),(0,2),(2,3)],
                          3:[(1,0)],4:[(1,0),(0,2)],5:[(1,3)],6:[(2,0)],
                          7:[(2,0),(0,1)],8:[(2,3)],9:[(3,1),(1,0)],10:[(3,1)],11:[(3,2)]
                          }
        """this will give MLU 1.0"""
        each_flow_edges = {0: [(0, 1)],
                           1: [(0, 2)], 
                           2: [(0, 1),(1,3),(0,2),(2,3)], 
                           3: [(1, 0)] ,
                           4: [(1,0),(0, 2)], 
                           5: [(1, 3)],
                           6: [(2,0)],
                           7: [(2,0),(0 ,1)], 
                           8: [(2, 3)] ,
                           9: [(3, 1),(1,0)], 
                           10: [(3,1)], 
                           11: [(3,2)] ,
                           12: [(0,2),(2,4)] ,
                           13: [(4,2),(2,0)] ,
                           14: [(2, 4)] , 
                           15: [(4, 2)] , 
                           16: [(4, 2),(2,0),(0,1)] , 
                           17: [(4, 2),(2,3)] , 
                           18: [(1,0),(0, 2),(2,4)] , 
                           19: [(3, 2),(2,4)]
                          }
        r = LpVariable(name="congestion_ratio")
#         for pr in self.lp_pairs:
#             for edge in self.lp_links:
#                 #if (edge[0],edge[1]) not in each_flow_edges[pr]:
#                 model +=(ratio[pr, edge[0], edge[1]] <=0.0)
        
        for pr in self.lp_pairs:
#            print("for pr %s we add"%(pr))
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1 , "flow_conservation_constr1_%d"%pr)

        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs]), "link_load_constr%d"%ei)
            model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

        model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return obj_r, solution
    
    def current_mlu_optimal_routing_mlu(self, tm_idx):
        
        
        tm = self.traffic_matrices[tm_idx]

        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]
            #print("demand from src %s to dst %s is %s"%(s,d,demands[i])) 

        model = LpProblem(name="routing")
        #print('this %s is our self.pair_links' %(self.pair_links)) 
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
        #print(ratio)
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)
        #print("this %s is self.lp_links"%(self.lp_links))
        #print("this %s is self.lp_nodes"%(self.lp_nodes))
        #print("this %s is self.lp_pairs"%(self.lp_pairs))
        r = LpVariable(name="congestion_ratio")
#         for pr in self.lp_pairs:
#             for edge in self.lp_links:
#                 if (edge[0],edge[1]) not in each_flow_edges[pr]:
#                     model +=(ratio[pr, edge[0], edge[1]] <=0.0)

        for pr in self.lp_pairs:
#            print("for pr %s we add"%(pr))
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1 , "flow_conservation_constr1_%d"%pr)

        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs]), "link_load_constr%d"%ei)
            model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

        model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()
        mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(tm_idx, solution, eval_delay=False)
           
            
        return mlu,solution
    def mlu_optimal_routing_mlu(self, tm_idx,look_ahead_window):
        mlus= []
        solutions = []
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            tm = self.traffic_matrices[traffic_m_indx]

            demands = {}
            for i in range(self.num_pairs):
                s, d = self.pair_idx_to_sd[i]
                demands[i] = tm[s][d]
                #print("demand from src %s to dst %s is %s"%(s,d,demands[i])) 

            model = LpProblem(name="routing")
            #print('this %s is our self.pair_links' %(self.pair_links)) 
            ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
            #print(ratio)
            link_load = LpVariable.dicts(name="link_load", indexs=self.links)
            #print("this %s is self.lp_links"%(self.lp_links))
            #print("this %s is self.lp_nodes"%(self.lp_nodes))
            #print("this %s is self.lp_pairs"%(self.lp_pairs))
            r = LpVariable(name="congestion_ratio")
    #         for pr in self.lp_pairs:
    #             for edge in self.lp_links:
    #                 if (edge[0],edge[1]) not in each_flow_edges[pr]:
    #                     model +=(ratio[pr, edge[0], edge[1]] <=0.0)

            for pr in self.lp_pairs:
    #            print("for pr %s we add"%(pr))
                model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1 , "flow_conservation_constr1_%d"%pr)

            for pr in self.lp_pairs:
                model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)

            for pr in self.lp_pairs:
                for n in self.lp_nodes:
                    if n not in self.pair_idx_to_sd[pr]:
                        model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))

            for e in self.lp_links:
                ei = self.link_sd_to_idx[e]
                #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
                model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs]), "link_load_constr%d"%ei)
                model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

            model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])

            model.solve(solver=GLPK(msg=False))
            assert LpStatus[model.status] == 'Optimal'

            obj_r = r.value()
            solution = {}
            for k in ratio:
                solution[k] = ratio[k].value()
            mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(traffic_m_indx, solution, eval_delay=False)
            mlus.append(mlu)
            solutions.append(solution)
        return mlus,solutions
    def mlu_routing_selected_paths_oblivious2(self,tm_idx,look_ahead_window,each_flow_edges):
        
        #print("we check mlu for tm_idx",tm_idx)
        tm_mlu_using_suggested_paths = []
        solutions = []
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            #for k in each_flow_edges.keys():
                #print('flow is ',k)
            for i in range(self.num_pairs):
                s, d = self.pair_idx_to_sd[i]
                #print('flow is ',s,d)
            #print('self.lp_pairs',self.lp_pairs)
            #print(each_flow_edges)
            import pdb
            #pdb.set_trace()

            #traffic_m_indx = tm_idx
            tm = self.traffic_matrices[traffic_m_indx]

            demands = {}
            for i in range(self.num_pairs):
                s, d = self.pair_idx_to_sd[i]
                demands[i] = tm[s][d]
                #demands[i] = 0
                #print("demand from src %s to dst %s is %s"%(s,d,demands[i])) 

            model = LpProblem(name="routing")
    #         print('this %s is our self.pair_links' %(self.pair_links)) 
            ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
            #print(ratio)
            link_load = LpVariable.dicts(name="link_load", indexs=self.links)
    #         print("this %s is self.lp_links"%(self.lp_links))
    #         print("this %s is self.lp_nodes"%(self.lp_nodes))
    #         print("this %s is self.lp_pairs"%(self.lp_pairs))
            import pdb

            r = LpVariable(name="congestion_ratio")
            for pr in self.lp_pairs:
                s, d = self.pair_idx_to_sd[pr]
                for edge in self.lp_links:
                    if (edge[0],edge[1]) not in each_flow_edges[(s,d)]:
                        model +=(ratio[pr, edge[0], edge[1]] ==0.0)

            for pr in self.lp_pairs:
    #            print("for pr %s we add"%(pr))
                #s, d = self.pair_idx_to_sd[pr]
                model += (lpSum([ratio[pr, e[0], e[1]] 
                                 for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - 
                          lpSum([ratio[pr, e[0], e[1]] 
                            for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1 ,
                          "flow_conservation_constr1_%d"%pr)

            for pr in self.lp_pairs:
                model += (lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) -
                        lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                          "flow_conservation_constr2_%d"%pr)

            for pr in self.lp_pairs:
                for n in self.lp_nodes:
                    if n not in self.pair_idx_to_sd[pr]:
                        model += (lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[1] == n]) -
                        lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                        "flow_conservation_constr3_%d_%d"%(pr,n))

            for e in self.lp_links:
                ei = self.link_sd_to_idx[e]
                #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
                model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] 
                    for pr in self.lp_pairs]), "link_load_constr%d"%ei)

                model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

            model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])

            model.solve(solver=GLPK(msg=False))
            assert LpStatus[model.status] == 'Optimal'

            obj_r = r.value()
            solution = {}
            for k in ratio:
                solution[k] = ratio[k].value()
        
            tm = self.traffic_matrices[traffic_m_indx]
            oblivious_mlu, oblivious_mlu_delay = self.eval_optimal_routing_mlu(traffic_m_indx, solution, eval_delay=False)
            tm_mlu_using_suggested_paths.append(oblivious_mlu)
            solutions.append(solution)
            #print("******* we got the solution ",drl_mlu)
        #return obj_r, solution 
        #print("here are the four mlu",tm_mlu_using_suggested_paths)
        return tm_mlu_using_suggested_paths,solutions
    def mlu_routing_selected_paths_oblivious(self,tm_idx,look_ahead_window,each_flow_edges):
        
        tm_mlu_using_suggested_paths = []
        solutions = []
        #for k in each_flow_edges.keys():
            #print('flow is ',k)
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            #print("these are the edges of flow %s %s "%((s,d),each_flow_edges[(s,d)]))
            #print('flow is ',s,d)
        #print('self.lp_pairs',self.lp_pairs)
        #print(each_flow_edges)
        import pdb
        #pdb.set_trace()
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            tm = self.traffic_matrices[traffic_m_indx]

            demands = {}
            for i in range(self.num_pairs):
                s, d = self.pair_idx_to_sd[i]
                demands[i] = tm[s][d]
                #demands[i] = 0
                #print("demand from src %s to dst %s is %s"%(s,d,demands[i])) 

            model = LpProblem(name="routing")
    #         print('this %s is our self.pair_links' %(self.pair_links)) 
            ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
            #print(ratio)
            link_load = LpVariable.dicts(name="link_load", indexs=self.links)
    #         print("this %s is self.lp_links"%(self.lp_links))
    #         print("this %s is self.lp_nodes"%(self.lp_nodes))
    #         print("this %s is self.lp_pairs"%(self.lp_pairs))
            import pdb
            
            r = LpVariable(name="congestion_ratio")
            for pr in self.lp_pairs:
                s, d = self.pair_idx_to_sd[pr]
                for edge in self.lp_links:
                    if (edge[0],edge[1]) not in each_flow_edges[(s,d)]:
                        model +=(ratio[pr, edge[0], edge[1]] ==0.0)

            for pr in self.lp_pairs:
    #            print("for pr %s we add"%(pr))
                #s, d = self.pair_idx_to_sd[pr]
                model += (lpSum([ratio[pr, e[0], e[1]] 
                                 for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - 
                          lpSum([ratio[pr, e[0], e[1]] 
                            for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1 ,
                          "flow_conservation_constr1_%d"%pr)

            for pr in self.lp_pairs:
                model += (lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) -
                        lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                          "flow_conservation_constr2_%d"%pr)

            for pr in self.lp_pairs:
                for n in self.lp_nodes:
                    if n not in self.pair_idx_to_sd[pr]:
                        model += (lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[1] == n]) -
                        lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                        "flow_conservation_constr3_%d_%d"%(pr,n))

            for e in self.lp_links:
                ei = self.link_sd_to_idx[e]
                #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
                model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] 
                    for pr in self.lp_pairs]), "link_load_constr%d"%ei)

                model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

            model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])

            model.solve(solver=GLPK(msg=False))
            assert LpStatus[model.status] == 'Optimal'

            obj_r = r.value()
            solution = {}
            for k in ratio:
                solution[k] = ratio[k].value()
            drl_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(traffic_m_indx, solution, eval_delay=False)
            tm_mlu_using_suggested_paths.append(drl_mlu)
            solutions.append(solution)
            #print("******* we got the solution ",drl_mlu)
        #return obj_r, solution 
        #print("here are the four mlu",tm_mlu_using_suggested_paths)
        return tm_mlu_using_suggested_paths,solutions
    def optimal_solution_for_robust_paths(self,tm_idx,look_ahead_window):
        #print("we check mlu for tm_idx",tm_idx)
        T= 0
        R=0
        all_paths = []
        for path in self.each_path_edges:
            if path not in all_paths:
                all_paths.append(path)
        for flow in self.lp_pairs:
            R+=1 
        R = max(R,self.max_moves)
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            T+=1
        all_flows = []
        new_each_flow_paths = {}
        for pr in self.lp_pairs:
            s, d = self.pair_idx_to_sd[pr]
            for path in self.each_flow_paths[(s,d)]:
                try:
                    new_each_flow_paths[pr].append(path)
                except:
                    new_each_flow_paths[pr]=[path]
            all_flows.append(pr)
#         for flow,paths in new_each_flow_paths.items():
#             print("flow %s paths %s"%(flow,paths))
        each_flow_paths = {}
        each_path_flow = {}
        for f,paths in new_each_flow_paths.items():
            each_flow_paths[f] = paths
            for path in paths:
                each_path_flow[path] = f
        tm_mlu_using_suggested_paths = []
        solutions = []

        import pdb
        #pdb.set_trace()
        demands = {}
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            tm = self.traffic_matrices[traffic_m_indx]
            for i in range(self.num_pairs):
                s, d = self.pair_idx_to_sd[i]
                try:
                    demands[traffic_m_indx][i] = tm[s][d]
                except:
                    demands[traffic_m_indx]={}
                    demands[traffic_m_indx][i] = tm[s][d]
                    
                #demands[i] = 0
                #print("demand from src %s to dst %s is %s"%(s,d,demands[i])) 
        time_links_load_indexes = []
        model = LpProblem(name="routing")
    #         print('this %s is our self.pair_links' %(self.pair_links)) 
        new_time_flow_paths_indexes = []
        time_link_tracker_id = 0
        each_t_link_index = {}
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            #print("these are pair links",self.pair_links)
            for link in self.links:
                time_links_load_indexes.append(time_link_tracker_id)
                #print("key is %s trafic_mx_id %s link %s"%((traffic_m_indx,link),traffic_m_indx,link))
                each_t_link_index[(traffic_m_indx,link)] = time_link_tracker_id
                time_link_tracker_id +=1
            for pr,paths in each_flow_paths.items():
                for path_id in paths:
                    new_time_flow_paths_indexes.append((traffic_m_indx,pr,path_id))
                #print("for time %s flow %s links are %s "%(traffic_m_indx,flow_links))
        import pdb
        
        ratio = LpVariable.dicts(name="ratio", indexs=new_time_flow_paths_indexes, lowBound=0, upBound=1)
        each_path_variable = LpVariable.dicts(name="pathflagh", indexs=self.each_path_edges.keys(),cat="Integer", lowBound=0, upBound=1)
        #print(ratio)
        
        link_load = LpVariable.dicts(name="time_link_load", indexs=time_links_load_indexes)
#         print("this %s is self.lp_links"%(self.lp_links))
#         print("this %s is self.lp_nodes"%(self.lp_nodes))
#         print("this %s is self.lp_pairs"%(self.lp_pairs))
        import pdb
        
        r = LpVariable(name="congestion_ratio")
#         for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
#             for pr in all_flows:
#                 for path in each_flow_paths[pr]:
#                 #for edge in self.lp_links:
# #                     if (edge[0],edge[1]) not in each_flow_edges[(s,d)]:
#                     if  each_path_variable[path] ==0:
#                         model +=(ratio[traffic_m_indx,pr, path] ==0.0)
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            for pr in self.lp_pairs:
                s, d = self.pair_idx_to_sd[pr]
                flow = (s,d)
    #            print("for pr %s we add"%(pr))
                #s, d = self.pair_idx_to_sd[pr]
                model += (lpSum([ratio[traffic_m_indx,pr, path] 
                                 for path in each_flow_paths[pr]]) == 1 ,
                          "flow_conservation_constr1_%d_%d"%(traffic_m_indx,pr))
        #print("the value of R is %s"%(R))
        model += (lpSum([each_path_variable[path] for path in all_paths]) == R,"all_paths")

#         for pr in self.lp_pairs:
#             for n in self.lp_nodes:
#                 if n not in self.pair_idx_to_sd[pr]:
#                     model += (lpSum([ratio[pr, e[0], e[1]] 
#                     for e in self.lp_links if e[1] == n]) -
#                     lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
#                     "flow_conservation_constr3_%d_%d"%(pr,n))

#         for e in self.lp_links:
#             ei = self.link_sd_to_idx[e]
#             print("e is %s and ei is %s "%(e,ei))
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            for e in self.lp_links:
                ei = self.link_sd_to_idx[e]
                #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
                #print("each_t_link_index ",each_t_link_index)
                time_link_indx = each_t_link_index[(traffic_m_indx,ei)]
#                 for pr in all_flows:
#                     for path in each_flow_paths[pr]:
#                         if e in each_path_edges[path]:
#                             print("time %s time_link_indx %s edge %s,%s was in path %s for flow %s"%(traffic_m_indx,time_link_indx,e,ei,each_path_edges[path],pr))
#                         else:
#                             print("time %s time_link_indx %s edge %s,%s was not in path %s for flow %s"%(traffic_m_indx,time_link_indx,e,ei,each_path_edges[path],pr))
#                         print("link load time link indx",link_load[time_link_indx])
                #print("thsi is link_load ***** ",link_load)
    
                model += (link_load[time_link_indx] == lpSum([demands[traffic_m_indx][pr]*ratio[traffic_m_indx,pr, path]
                    for pr in all_flows  for path in each_flow_paths[pr] if (e in self.each_path_edges[path])]), "link_load_constr%d_%d"%(traffic_m_indx,ei))

                model += (link_load[time_link_indx] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d%d"%(traffic_m_indx,ei))

        model += r + OBJ_EPSILON*lpSum([link_load[t_e] for t_e in time_links_load_indexes])/T

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        optimal_robust_paths = []
        each_flow_robust_paths_edges={}
        robust_path_counter = 0
        
        optimal_robust_paths = []
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            for pr,paths in each_flow_paths.items():
                rate_sum = 0
                flow =  self.pair_idx_to_sd[pr]
                for path_id in paths:
                    rate_sum+= ratio[traffic_m_indx,pr, path_id].value()
                    if ratio[traffic_m_indx,pr, path_id].value()>0:
                        this_path_edges = self.each_path_edges[path_id]
                        #print("for time %s flow %s path %s edges are %s"%(traffic_m_indx,flow,path,this_path_edges))
                        if path_id not in optimal_robust_paths:
                            optimal_robust_paths.append(path_id)
                        for edge in this_path_edges:
                            try:
                                if edge not in each_flow_robust_paths_edges[flow]:
                                    each_flow_robust_paths_edges[flow].append(edge)
                            except:
                                try:
                                    each_flow_robust_paths_edges[flow].append(edge)
                                except:
                                    each_flow_robust_paths_edges[flow]=[edge]
                    
                    
                    
                    
#                 print("1: at time %s for flow %s the sum of rates of all paths is %s"%(traffic_m_indx,pr,rate_sum))     
                          
              
        
#         for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
#             for pr in self.lp_pairs:
#                 s, d = self.pair_idx_to_sd[pr]
#                 flow = (s,d)
#                 rate_sum  = 0
#                 for path in each_flow_paths[pr]:
#                     if each_path_variable[path].value()==1:
#                         rate_sum = rate_sum+ratio[traffic_m_indx,pr, path].value() 
#                         this_path_edges = each_path_edges[path]
#                         #print("for time %s flow %s path %s edges are %s"%(traffic_m_indx,flow,path,this_path_edges))
#                         if path not in optimal_robust_paths:
#                             optimal_robust_paths.append(path)
#                         for edge in this_path_edges:
#                             try:
#                                 if edge not in each_flow_robust_paths_edges[flow]:
#                                     each_flow_robust_paths_edges[flow].append(edge)
#                             except:
#                                 try:
#                                     each_flow_robust_paths_edges[flow].append(edge)
#                                 except:
#                                     each_flow_robust_paths_edges[flow]=[edge]
#                 print("at time %s for flow %s the sum of rates of all paths is %s"%(traffic_m_indx,flow,rate_sum))
                                  
        import pdb
        
#         for pr,paths in each_flow_paths.items():
#             flow = self.pair_idx_to_sd[pr]
#             if flow not in each_flow_robust_paths_edges:
#                 print("we do not have even one path for flow!!!! ",flow)
        #print("selected path numbers ",len(optimal_robust_paths))
        #for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            #print("time is ",traffic_m_indx)
        #print(each_flow_robust_paths_edges)
        #print(each_flow_robust_paths_edges[(0,1)])
        #pdb.set_trace()
        optimal_mlus,optimal_solutions = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
                                                               each_flow_robust_paths_edges)
        
        
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            solution = {}
            for k in ratio:
                
                #print(ratio)
                time_of_key  = k[0]
                #print("This is traffic indx %s and k is %s and time of key is %s"%(traffic_m_indx,k,time_of_key))
                if time_of_key==traffic_m_indx:
                    solution[k] = ratio[k].value()
            optimal_mlu, optimal_mlu_delay = self.eval_optimal_robust_paths_routing_mlu(traffic_m_indx, solution, each_flow_paths,eval_delay=False)
            #print("we got the mlu %s for time %s "%(optimal_mlu,traffic_m_indx))
            tm_mlu_using_suggested_paths.append(optimal_mlu)
            solutions.append(solution)
            #pdb.set_trace()
        #print("******* we got the solution ",drl_mlu)
        #return obj_r, solution 
        #print("here are the four mlu",tm_mlu_using_suggested_paths)
        #print("mlu using robust paths without hashing %s and with rehashing %s"%
#               (sum(optimal_mlus)/len(optimal_mlus),sum(tm_mlu_using_suggested_paths)/len(tm_mlu_using_suggested_paths)))
#         return tm_mlu_using_suggested_paths,solutions
        number_of_selected_paths = 0
        for path in all_paths:
            number_of_selected_paths+= each_path_variable[path].value()
        optimal_robust_paths.sort()
#         for path in optimal_robust_paths:
#             print("for optimal we have %s their flag %s path %s "%(len(optimal_robust_paths),number_of_selected_paths,path))
        #return optimal_mlus,optimal_solutions,tm_mlu_using_suggested_paths,solutions
        return optimal_mlus,optimal_solutions,optimal_mlus,optimal_solutions
        if sum(optimal_mlus)/len(optimal_mlus)<sum(tm_mlu_using_suggested_paths)/len(tm_mlu_using_suggested_paths):
            return optimal_mlus,optimal_solutions
        else:
            return tm_mlu_using_suggested_paths,solutions
        
    def mlu_routing_selected_paths(self,tm_idx,look_ahead_window,each_flow_edges):
        #print("we check mlu for tm_idx",tm_idx)
        tm_mlu_using_suggested_paths = []
        solutions = []
        #for k in each_flow_edges.keys():
            #print('flow is ',k)
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            #print("these are the edges of flow %s %s "%((s,d),each_flow_edges[(s,d)]))
            #print('flow is ',s,d)
        #print('self.lp_pairs',self.lp_pairs)
        #print(each_flow_edges)
        import pdb
        #pdb.set_trace()
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            tm = self.traffic_matrices[traffic_m_indx]

            demands = {}
            for i in range(self.num_pairs):
                s, d = self.pair_idx_to_sd[i]
                demands[i] = tm[s][d]
                #demands[i] = 0
                #print("demand from src %s to dst %s is %s"%(s,d,demands[i])) 

            model = LpProblem(name="routing")
    #         print('this %s is our self.pair_links' %(self.pair_links)) 
            ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
            #print(ratio)
            link_load = LpVariable.dicts(name="link_load", indexs=self.links)
    #         print("this %s is self.lp_links"%(self.lp_links))
    #         print("this %s is self.lp_nodes"%(self.lp_nodes))
    #         print("this %s is self.lp_pairs"%(self.lp_pairs))
            import pdb
            
            r = LpVariable(name="congestion_ratio")
            for pr in self.lp_pairs:
                s, d = self.pair_idx_to_sd[pr]
                for edge in self.lp_links:
                    if (edge[0],edge[1]) not in each_flow_edges[(s,d)]:
                        model +=(ratio[pr, edge[0], edge[1]] ==0.0)

            for pr in self.lp_pairs:
    #            print("for pr %s we add"%(pr))
                #s, d = self.pair_idx_to_sd[pr]
                model += (lpSum([ratio[pr, e[0], e[1]] 
                                 for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - 
                          lpSum([ratio[pr, e[0], e[1]] 
                            for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1 ,
                          "flow_conservation_constr1_%d"%pr)

            for pr in self.lp_pairs:
                model += (lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) -
                        lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                          "flow_conservation_constr2_%d"%pr)

            for pr in self.lp_pairs:
                for n in self.lp_nodes:
                    if n not in self.pair_idx_to_sd[pr]:
                        model += (lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[1] == n]) -
                        lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                        "flow_conservation_constr3_%d_%d"%(pr,n))

            for e in self.lp_links:
                ei = self.link_sd_to_idx[e]
                #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
                model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] 
                    for pr in self.lp_pairs]), "link_load_constr%d"%ei)

                model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

            model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])

            model.solve(solver=GLPK(msg=False))
            assert LpStatus[model.status] == 'Optimal'

            obj_r = r.value()
            solution = {}
            for k in ratio:
                solution[k] = ratio[k].value()
            drl_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(traffic_m_indx, solution, eval_delay=False)
            tm_mlu_using_suggested_paths.append(drl_mlu)
            solutions.append(solution)
            #print("******* we got the solution ",drl_mlu)
        #return obj_r, solution 
        #print("here are the four mlu",tm_mlu_using_suggested_paths)
        return tm_mlu_using_suggested_paths,solutions
    def mlu_routing_selected_paths_fixed_rates(self,tm_idx,look_ahead_window,each_flow_edges):
        #print("we check mlu for tm_idx",tm_idx)
        tm_mlu_using_suggested_paths = []
        solutions = []
        #for k in each_flow_edges.keys():
            #print('flow is ',k)
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            #print("these are the edges of flow %s %s "%((s,d),each_flow_edges[(s,d)]))
            #print('flow is ',s,d)
        #print('self.lp_pairs',self.lp_pairs)
        #print(each_flow_edges)
        import pdb
        #pdb.set_trace()

        tm = self.traffic_matrices[tm_idx]

        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]
            #demands[i] = 0
            #print("demand from src %s to dst %s is %s"%(s,d,demands[i])) 

        model = LpProblem(name="routing")
#         print('this %s is our self.pair_links' %(self.pair_links)) 
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
        #print(ratio)
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)
#         print("this %s is self.lp_links"%(self.lp_links))
#         print("this %s is self.lp_nodes"%(self.lp_nodes))
#         print("this %s is self.lp_pairs"%(self.lp_pairs))
        import pdb

        r = LpVariable(name="congestion_ratio")
        for pr in self.lp_pairs:
            s, d = self.pair_idx_to_sd[pr]
            for edge in self.lp_links:
                if (edge[0],edge[1]) not in each_flow_edges[(s,d)]:
                    model +=(ratio[pr, edge[0], edge[1]] ==0.0)

        for pr in self.lp_pairs:
#            print("for pr %s we add"%(pr))
            #s, d = self.pair_idx_to_sd[pr]
            model += (lpSum([ratio[pr, e[0], e[1]] 
                             for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - 
                      lpSum([ratio[pr, e[0], e[1]] 
                        for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1 ,
                      "flow_conservation_constr1_%d"%pr)

        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] 
                    for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) -
                    lpSum([ratio[pr, e[0], e[1]] 
                    for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                      "flow_conservation_constr2_%d"%pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] 
                    for e in self.lp_links if e[1] == n]) -
                    lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                    "flow_conservation_constr3_%d_%d"%(pr,n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            #print("for edge %s ei %s capacity is %s "%(e,ei,self.link_capacities[ei]))
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] 
                for pr in self.lp_pairs]), "link_load_constr%d"%ei)

            model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

        model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()
        for traffic_m_indx in range(tm_idx,tm_idx+look_ahead_window-1):
            drl_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(traffic_m_indx, solution, eval_delay=False)
            tm_mlu_using_suggested_paths.append(drl_mlu)
            solutions.append(solution)
            #print("******* we got the solution ",drl_mlu)
        #return obj_r, solution 
        #print("here are the four mlu",tm_mlu_using_suggested_paths)
        return tm_mlu_using_suggested_paths,solutions
    def eval_optimal_robust_paths_routing_mlu(self,tm_idx, solution,each_flow_paths, eval_delay=False):
        optimal_link_loads = np.zeros((self.num_links))
        eval_tm = self.traffic_matrices[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                for path in each_flow_paths[i]:
                    if e in self.each_path_edges[path]:
                        #print("for src %s dst %s  e %s we have demand %s and solution %s "%(s,d,e,demand,solution[i, e[0], e[1]]))
                        optimal_link_loads[link_idx] += demand*solution[tm_idx,i, path]
        
        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        
        delay = 0
        return optimal_max_utilization,delay
    def eval_optimal_routing_mlu(self, tm_idx, solution, eval_delay=False):
        optimal_link_loads = np.zeros((self.num_links))
        eval_tm = self.traffic_matrices[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                #print("for src %s dst %s  e %s we have demand %s and solution %s "%(s,d,e,demand,solution[i, e[0], e[1]]))
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]
        
        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            optimal_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))
                        
        return optimal_max_utilization, delay

    def optimal_routing_mlu_critical_pairs(self, tm_idx, critical_pairs):
        tm = self.traffic_matrices[tm_idx]

        pairs = critical_pairs

        demands = {}
        background_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            #background link load
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(background_link_loads, tm[s][d], s, d)
            else:
                demands[i] = tm[s][d]

        model = LpProblem(name="routing")
        
        pair_links = [(pr, e[0], e[1]) for pr in pairs for e in self.lp_links] 
        ratio = LpVariable.dicts(name="ratio", indexs=pair_links, lowBound=0, upBound=1)
        
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)

        r = LpVariable(name="congestion_ratio")

        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, "flow_conservation_constr1_%d"%pr)

        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)

        for pr in pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == background_link_loads[ei] + lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in pairs]), "link_load_constr%d"%ei)
            model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)

        model += r + OBJ_EPSILON*lpSum([link_load[ei] for ei in self.links])

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return obj_r, solution

    def eval_critical_flow_and_ecmp(self, tm_idx, critical_pairs, solution, eval_delay=False):
        eval_tm = self.traffic_matrices[tm_idx]
        eval_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(eval_link_loads, eval_tm[s][d], s, d)
            else:
                demand = eval_tm[s][d]
                for e in self.lp_links:
                    link_idx = self.link_sd_to_idx[e]
                    eval_link_loads[link_idx] += eval_tm[s][d]*solution[i, e[0], e[1]]

        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))
        
        return eval_max_utilization, delay

    def optimal_routing_delay(self, tm_idx):
        assert tm_idx in self.load_multiplier, (tm_idx)
        tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        model = LpProblem(name="routing")
     
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name="link_load", indexs=self.links)

        f = LpVariable.dicts(name="link_cost", indexs=self.links)

        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, "flow_conservation_constr1_%d"%pr)

        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs]), "link_load_constr%d"%ei)
            model += (f[ei] * self.link_capacities[ei] >= link_load[ei], "cost_constr1_%d"%ei)
            model += (f[ei] >= 3 * link_load[ei] / self.link_capacities[ei] - 2/3, "cost_constr2_%d"%ei)
            model += (f[ei] >= 10 * link_load[ei] / self.link_capacities[ei] - 16/3, "cost_constr3_%d"%ei)
            model += (f[ei] >= 70 * link_load[ei] / self.link_capacities[ei] - 178/3, "cost_constr4_%d"%ei)
            model += (f[ei] >= 500 * link_load[ei] / self.link_capacities[ei] - 1468/3, "cost_constr5_%d"%ei)
            model += (f[ei] >= 5000 * link_load[ei] / self.link_capacities[ei] - 16318/3, "cost_constr6_%d"%ei)
       
        model += lpSum(f[ei] for ei in self.links)

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return solution

    def eval_optimal_routing_delay(self, tm_idx, solution):
        optimal_link_loads = np.zeros((self.num_links))
        assert tm_idx in self.load_multiplier, (tm_idx)
        eval_tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]
        
        optimal_delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_delay
    def get_suggested_paths_by_rl(self,actions):
        
        for flow,paths in self.each_flow_paths.items():
            covered_flow = False
            for path_id in paths:
                #path_id = self.each_path_id[path]
                if not covered_flow:
                    if path_id in actions:
                        covered_flow =True 
            if not covered_flow:
                #print("flow  did not have any candidate path in chosen paths",flow)
                flow_shortest_path = self.each_flow_shortest_path[flow]
                #print(flow_shortest_path)
                #print(each_path_id)
                path_id = self.each_path_id[tuple(flow_shortest_path)]

                actions = np.append(actions, path_id)
#         print("these are the paths of flow ",self.each_flow_paths[(0,4)])
#         for flow in self.each_flow_paths:
#             if flow not in self.each_flow_shortest_path:
#                 print("the flow %s does not have a shortest path!!!!"%(flow))
        each_flow_edges = {}
        for flow in self.each_flow_shortest_path:
            #print("for flow ",flow)
            for path in actions:
                #path = each_id_path[path_id]
                if path in self.each_flow_paths[flow]:
                    #print("we are going to add these edges ",path_id,path)
                    path_edges = self.each_path_edges[path]
                    for edge in path_edges:
                        try:
                            if edge not in each_flow_edges[flow]:
                                each_flow_edges[flow].append(edge)
                                each_flow_edges[flow].append((edge[1],edge[0]))
                        except:
                            each_flow_edges[flow]=[edge]
                            each_flow_edges[flow].append((edge[1],edge[0]))
        return each_flow_edges
    
    def get_set_of_all_paths(self,tm_idx,look_ahead_window,each_flow_paths,each_path_id,each_id_path,each_flow_shortest_path,each_flow_oblivious_paths,topology_file,config):
        target_times = []
        each_flow_paths2 = {}
        for flow,paths in each_flow_paths.items():
            each_flow_paths2[flow] = list(paths)
        
        for flow,path in each_flow_shortest_path.items():
            #print("these are shortest paths ",paths)
            path = tuple(path)
            if path not in each_flow_paths2[flow]:
                each_flow_paths[flow].add(path)
        
#         for flow,paths in each_flow_oblivious_paths.items():
            
#             for path in paths:
# #                 path = tuple(path)
#                 path = path[0]
                
#                 if path not in each_flow_paths2[flow]:
#                     each_flow_paths[flow].add(path)
        """we now add the edges for each flow from the selected actions"""
        each_path_edges = {}
        for flow,paths in each_flow_paths2.items():
            for path in paths:
                path_id = each_path_id[path]
                for node_indx in range(len(path)-1):
                    try:
                        if (path[node_indx],path[node_indx+1]) not in each_path_edges[path_id]:
                            each_path_edges[path_id].append((path[node_indx],path[node_indx+1]))
                            each_path_edges[path_id].append((path[node_indx+1],path[node_indx]))
                    except:
                        each_path_edges[path_id]=[(path[node_indx],path[node_indx+1])]
                        each_path_edges[path_id].append((path[node_indx+1],path[node_indx]))
        
#         for flow,edges in each_flow_edges.items():
#             print("this flow %s has these edges %s"%(flow,edges))
#         print("and this is our max moves",self.max_moves)
        new_each_flow_paths = {}
        for flow,paths in each_flow_paths2.items():
            new_paths = []
            for path in paths:
                path_id = each_path_id[path]
                new_paths.append(path_id)
            new_each_flow_paths[flow] = new_paths
        return new_each_flow_paths,each_path_edges
    def get_hindsight_set_of_paths(self,tm_idx,look_ahead_window,each_flow_paths,each_path_id,each_id_path,each_flow_shortest_path,topology_file,config):
        target_times = []
        for t in range(tm_idx,tm_idx+look_ahead_window):
            target_times.append(t)
        each_path_counter = {}
        #print('topology is ',topology_file)
        with open(config.each_topology_each_t_each_f_paths) as file:
            lines = file.readlines()
            for line in lines:
                if topology_file in line:#ATT_topology_file_modified:189:1->20: ['1', '13', '9', '4', '20'],['1', '9', '4', '20']
                    time = line.split(":")[1]
                    if int(time) in target_times:
                        flow_str = line.split(":")[2]
                        flow = flow_str.split("->")
                        flow = (int(flow[0]),int(flow[1]))
                        paths = line.split(flow_str+":")[1]
                        paths = paths.strip()
                        paths = "\""+str(paths)+"\""
                        paths = ast.literal_eval(paths)
                        #print('time %s flow %s paths %s'%(time,flow, paths))
                        #print(type(paths),len(paths))
                        if '],[' in paths:
                            paths = ast.literal_eval(paths)
                        else:
                            paths = ast.literal_eval(paths)
                            new_path = []
                            for p in paths:
                                new_path.append(p)
                            paths = [new_path]
                        #print('2time %s flow %s paths %s'%(time,flow, paths))
                        #print(type(paths),len(paths))
                        for p in paths:
                            p = tuple(p)
                            new_p= []
                            for node in p:
                                new_p.append(int(node))
                            p = tuple(new_p)
                            try:
                                each_path_counter[p]+=1
                            except:
                                each_path_counter[p]=1
        repeated_times = []
        for path,counter in   each_path_counter.items():
            if counter not in repeated_times:
                repeated_times.append(counter)
        repeated_times.sort()
        top_selected_path_ids = []
        top_selected_counters = repeated_times[-self.max_moves:]
        for p,counter in each_path_counter.items():
            if counter in top_selected_counters:
                if len(top_selected_path_ids)<self.max_moves:
                    path_id = each_path_id[p]
                    top_selected_path_ids.append(path_id)
        
        for flow,paths in each_flow_paths.items():
            covered_flow = False
            for path in paths:
                path_id = each_path_id[path]
                if not covered_flow:
                    if path_id in top_selected_path_ids:
                        covered_flow =True 
            if not covered_flow:
                #print("flow  did not have any candidate path in chosen paths",flow)
                flow_shortest_path = each_flow_shortest_path[flow]
                #print(flow_shortest_path)
                #print(each_path_id)
                path_id = self.each_path_id[tuple(flow_shortest_path)]
                
                top_selected_path_ids.append(path_id)
                #print("we added path id %s for safety"%(path_id))
        
        """end of safe online learning section"""
        
#         for path_id in actions:
#             print("we have chosen and safed action %s %s"%(path_id,len(actions)))
            
        each_flow_selected_paths = {}
        """we now add the edges for each flow from the selected actions"""
        each_flow_edges = {}
        for flow in each_flow_shortest_path:
            for path_id in top_selected_path_ids:
                path = each_id_path[path_id]
                if path in each_flow_paths[flow]:
                    try:
                        each_flow_selected_paths[flow].append(path)
                    except:
                        each_flow_selected_paths[flow]=[path]
                    for node_indx in range(len(path)-1):
                        try:
                            if (path[node_indx],path[node_indx+1]) not in each_flow_edges[flow]:
                                each_flow_edges[flow].append((path[node_indx],path[node_indx+1]))
                                each_flow_edges[flow].append((path[node_indx+1],path[node_indx]))
                        except:
                            each_flow_edges[flow]=[(path[node_indx],path[node_indx+1])]
                            each_flow_edges[flow].append((path[node_indx+1],path[node_indx]))
        
#         for flow,edges in each_flow_edges.items():
#             print("this flow %s has these edges %s"%(flow,edges))
#         print("and this is our max moves",self.max_moves)
        return each_flow_edges
# In[ ]:
    def set_paths_info(self,env,topology_name,each_topology_each_t_each_f_paths):
    
        self.each_path_edges = {}
        self.each_flow_paths = {}
        self.each_flow_shortest_path ={}
        self.each_path_id={}
        self.each_id_path={}
        all_the_paths = set([])
        each_t_paths = {}
        #print('topology is ',topology_name)
        with open(each_topology_each_t_each_f_paths) as file:
            lines = file.readlines()
            for line in lines:
                if topology_name in line:#ATT_topology_file_modified:189:1->20: ['1', '13', '9', '4', '20'],['1', '9', '4', '20']
                    time = line.split(":")[1]
                    flow_str = line.split(":")[2]
                    flow = flow_str.split("->")
                    flow = (int(flow[0]),int(flow[1]))
                    paths = line.split(flow_str+":")[1]
                    paths = paths.strip()
                    paths = "\""+str(paths)+"\""
                    paths = ast.literal_eval(paths)
                    #print('time %s flow %s paths %s'%(time,flow, paths))
                    #print(type(paths),len(paths))
                    if '],[' in paths:
                        paths = ast.literal_eval(paths)
                    else:
                        paths = ast.literal_eval(paths)
                        new_path = []
                        for p in paths:
                            new_path.append(p)
                        paths = [new_path]
                    #print('2time %s flow %s paths %s'%(time,flow, paths))
                    #print(type(paths),len(paths))
                    for p in paths:
                        p = tuple(p)
                        new_p= []
                        for node in p:
                            new_p.append(int(node))
                        p = tuple(new_p)

                        try:
                            each_t_paths[time].add(p)
                        except:
                            each_t_paths[time] = set([p])
                        all_the_paths.add(p)

        
        
        
        path_counter = 0
        each_path_id = {}
        set_of_times = set([])
        each_flow_path_usage = {}
        all_the_paths = set([])
        each_t_paths = {}
        each_flow_paths = {}
        #print('topology is ',topology_name)
        with open(each_topology_each_t_each_f_paths) as file:
            lines = file.readlines()
            for line in lines:
                if topology_name in line:#ATT_topology_file_modified:189:1->20: ['1', '13', '9', '4', '20'],['1', '9', '4', '20']
                    time = line.split(":")[1]
                    flow_str = line.split(":")[2]
                    flow = flow_str.split("->")
                    flow = (int(flow[0]),int(flow[1]))
                    set_of_times.add(time)
                    paths = line.split(flow_str+":")[1]
                    paths = paths.strip()
                    paths = "\""+str(paths)+"\""
                    paths = ast.literal_eval(paths)
                    #print('time %s flow %s paths %s'%(time,flow, paths))
                    #print(type(paths),len(paths))
                    if '],[' in paths:
                        paths = ast.literal_eval(paths)
                    else:
                        paths = ast.literal_eval(paths)
                        new_path = []
                        for p in paths:
                            new_path.append(p)
                        paths = [new_path]
                    #print('2time %s flow %s paths %s'%(time,flow, paths))
                    #print(type(paths),len(paths))
                    for p in paths:
                        p = tuple(p)
                        new_p= []
                        for node in p:
                            new_p.append(int(node))
                        p = tuple(new_p)
                        try:
                            each_t_paths[time].add(p)
                        except:
                            each_t_paths[time] = set([p])
                            
                        try:
                            each_flow_paths[flow].add(p)
                        except:
                            each_flow_paths[flow] =set([p])
                        try:
                            each_t_paths[time].add(p)
                        except:
                            each_t_paths[time] = set([p])
                        all_the_paths.add(p)
                        #print('this is a path ',p)
                        
                        
        paths_number = []
        for time, paths in each_t_paths.items():
            #print('for time %s we had %s paths'%(time,len(paths)))
            paths_number.append(len(paths))
        avg_paths_per_time = int((sum(paths_number)/len(paths_number)))
        
        each_path_id = {}
        id_counter = 0
        each_id_path = {}
        covered_paths = []
        for flow,paths in each_flow_paths.items():
            for p in paths:
                #print("theis was path ",p)
                covered_paths.append(tuple(p))
                self.each_path_id[tuple(p)] = id_counter
                self.each_id_path[id_counter] = tuple(p)
                id_counter+=1
        self.each_flow_shortest_path = env.topology.get_each_flow_shortest_paths()
        for flow,paths in self.each_flow_shortest_path.items():
            
            #print("this is the path ",tuple(paths))
            if tuple(paths) not in covered_paths:
                each_flow_paths[flow].add(tuple(paths))
                covered_paths.append(tuple(paths))
                self.each_path_id[tuple(paths)] = id_counter
                self.each_id_path[id_counter] = tuple(paths)
                id_counter+=1
        
        
        for flow, paths in each_flow_paths.items():
            for path in paths:
                path_id = self.each_path_id[path]
                try:
                    self.each_flow_paths[flow].add(path_id)
                except:
                    self.each_flow_paths[flow]=set([path_id])
                for node_indx in range(len(path)-1):
                    try:
                        if (path[node_indx],path[node_indx+1]) not in self.each_path_edges[path_id]:
                            self.each_path_edges[path_id].append((path[node_indx],path[node_indx+1]))
                            self.each_path_edges[path_id].append((path[node_indx+1],path[node_indx]))
                    except:
                        self.each_path_edges[path_id]=[(path[node_indx],path[node_indx+1])]
                        self.each_path_edges[path_id].append((path[node_indx+1],path[node_indx]))
                
        
    def get_each_flow_paths_and_path_id(self,env,config,topology_name):
        """in this function, we get the B=4 most used paths by each flow

        we get all the used path by each flow over all times and get top 4 of them"""
        path_counter = 0
        each_path_id = {}
        set_of_times = set([])
        each_flow_path_usage = {}
        all_the_paths = set([])
        each_t_paths = {}
        each_flow_paths = {}
        #print('topology is ',topology_name)
        with open(config.each_topology_each_t_each_f_paths) as file:
            lines = file.readlines()
            for line in lines:
                if topology_name in line:#ATT_topology_file_modified:189:1->20: ['1', '13', '9', '4', '20'],['1', '9', '4', '20']
                    time = line.split(":")[1]
                    flow_str = line.split(":")[2]
                    flow = flow_str.split("->")
                    flow = (int(flow[0]),int(flow[1]))
                    set_of_times.add(time)
                    paths = line.split(flow_str+":")[1]
                    paths = paths.strip()
                    paths = "\""+str(paths)+"\""
                    paths = ast.literal_eval(paths)
                    #print('time %s flow %s paths %s'%(time,flow, paths))
                    #print(type(paths),len(paths))
                    if '],[' in paths:
                        paths = ast.literal_eval(paths)
                    else:
                        paths = ast.literal_eval(paths)
                        new_path = []
                        for p in paths:
                            new_path.append(p)
                        paths = [new_path]
                    #print('2time %s flow %s paths %s'%(time,flow, paths))
                    #print(type(paths),len(paths))
                    for p in paths:
                        p = tuple(p)
                        new_p= []
                        for node in p:
                            new_p.append(int(node))
                        p = tuple(new_p)
                        try:
                            each_flow_paths[flow].add(p)
                        except:
                            each_flow_paths[flow] =set([p])
                        try:
                            each_t_paths[time].add(p)
                        except:
                            each_t_paths[time] = set([p])
                        all_the_paths.add(p)
                        #print('this is a path ',p)
        each_path_id = {}
        id_counter = 0
        each_id_path = {}
        covered_paths = []
        for flow,paths in each_flow_paths.items():
            for p in paths:
                #print("theis was path ",p)
                covered_paths.append(tuple(p))
                each_path_id[tuple(p)] = id_counter
                each_id_path[id_counter] = tuple(p)
                id_counter+=1
        each_flow_shortest_paths = env.topology.get_each_flow_shortest_paths()
        for flow,paths in each_flow_shortest_paths.items():
            
            #print("this is the path ",tuple(paths))
            if tuple(paths) not in covered_paths:
                covered_paths.append(tuple(paths))
                each_path_id[tuple(paths)] = id_counter
                each_id_path[id_counter] = tuple(paths)
                id_counter+=1
        return each_flow_paths,each_path_id,each_id_path


# In[ ]:



class CFRRL_Game(Game):
    def __init__(self, config, env,commitment_window,look_ahead_window,path_counter,avg_paths_per_t, random_seed=1000):
        super(CFRRL_Game, self).__init__(config, env, random_seed)
        
        self.project_name = config.project_name
        #env.num_pairs = self.num_pairs
        self.action_dim = path_counter
        #print("self.action_dim is %s path_counter is %s "%(self.action_dim,path_counter))
        self.max_moves = int(self.action_dim * (config.max_moves / 100.))
        self.max_moves = avg_paths_per_t
        #print("self.max_moves %s self.action_dim %s self.max_moves %s self.action_dim %s"
       #       %(self.max_moves , self.action_dim, self.max_moves, self.action_dim))
        #assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)
        
        self.tm_history = 1
        self.tm_history=commitment_window
        self.tm_indexes = np.arange(self.tm_history-1, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)
        
        if config.method == 'pure_policy':
            self.baseline = {}

        self.generate_inputs(normalization=True)
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)
        print('Max moves :', self.max_moves)
        import pdb
        print('whole shapes ',self.normalized_traffic_matrices.shape)
        print('avg_paths_per_t',avg_paths_per_t)
        #pdb.set_trace()
        """two new variables"""
        
        self.commitment_window =commitment_window
        self.look_ahead_window = look_ahead_window

    def get_state(self, tm_idx):
        idx_offset = self.tm_history - 1
        return self.normalized_traffic_matrices[tm_idx-idx_offset]
    def reward2(self, tm_idx,all_tms,look_ahead_window,each_flow_edges,topology_file,printing_flag):
        
#         _, solution = self.optimal_routing_mlu(tm_idx)
#         optimal_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(tm_idx, solution, eval_delay=False)
#         print("here is the optimal mlu",optimal_mlu)
#         _, solution = self.optimal_routing_mlu_default(tm_idx)
#         optimal_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(tm_idx, solution, eval_delay=False)
#         print("here is the optimal mlu 2",optimal_mlu)
#         mlu, _ = self.optimal_routing_mlu_critical_pairs(tm_idx, actions)
        
#         print("here is the optimal mlu using critical flow rerouting",optimal_mlu)
        
        drl_mlus,solutions = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,each_flow_edges)
        drl_mlu = sum(drl_mlus)/len(drl_mlus)
        #print("this is the avg mlu using the suggested paths ",drl_mlu)
        
        #each_flow_edges_hindsight_approach = self.get_hindsight_set_of_paths(tm_idx,look_ahead_window,each_flow_paths,each_path_id,each_id_path,each_flow_shortest_path,topology_file)
        #hindsight_mlu = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,each_flow_edges_hindsight_approach)
        mlu_optimal_mlus,mlu_optimal_solutions = self.mlu_optimal_routing_mlu(tm_idx,look_ahead_window)
        mlu_optimal_mlu = sum(mlu_optimal_mlus)/len(mlu_optimal_mlus)
        reward = mlu_optimal_mlu/drl_mlu
        if printing_flag:
            print("for tm_idx point %s from %s rl is %s mlu optimal approach is %s and the reward is %s"%(tm_idx,all_tms,drl_mlu,mlu_optimal_mlu,reward))
        return reward
        import pdb
        pdb.set_trace()
        optimal_mlu = self.get_optimal_avg_mlu_hindsight(tm_idx,look_ahead_window)
        
    def reward(self, tm_idx, actions):
        mlu, _ = self.optimal_routing_mlu_critical_pairs(tm_idx, actions)

        reward = 1 / mlu

        return reward
    def get_all_trainig_epochs(self,commitment_window,look_ahead_window):
        indx = 1
        epochs = []
        epochs.append(1)
        epochs.append(9)
        while(max(epochs)<709):
            epochs.append(max(epochs)+30)
        while(max(epochs)<1999):
            epochs.append(max(epochs)+300)
        
        
        return list(epochs)
    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]
        
        #print(reward, (total_v/cnt))

        return reward - (total_v/cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)

    def evaluate(self, tm_idx, actions=None, ecmp=True, eval_delay=False):
        
        _, solution = self.optimal_routing_mlu(tm_idx)
        optimal_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(tm_idx, solution, eval_delay=eval_delay)
        #print("for tm_idx %s we have mlu %s"%(tm_idx,optimal_mlu))
        if ecmp:
            ecmp_mlu, ecmp_delay = self.eval_ecmp_traffic_distribution(tm_idx, eval_delay=eval_delay)
        
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, actions)
        mlu, delay = self.eval_critical_flow_and_ecmp(tm_idx, actions, solution, eval_delay=eval_delay)

        crit_topk = self.get_critical_topK_flows(tm_idx)
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, crit_topk)
        crit_mlu, crit_delay = self.eval_critical_flow_and_ecmp(tm_idx, crit_topk, solution, eval_delay=eval_delay)

        topk = self.get_topK_flows(tm_idx, self.lp_pairs)
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, topk)
        topk_mlu, topk_delay = self.eval_critical_flow_and_ecmp(tm_idx, topk, solution, eval_delay=eval_delay)


        norm_mlu = optimal_mlu / mlu
        line = str(tm_idx) + ', ' + str(norm_mlu) + ', ' + str(mlu) + ', ' 
        
        norm_crit_mlu = optimal_mlu / crit_mlu
        line += str(norm_crit_mlu) + ', ' + str(crit_mlu) + ', ' 

        norm_topk_mlu = optimal_mlu / topk_mlu
        line += str(norm_topk_mlu) + ', ' + str(topk_mlu) + ', ' 

        if ecmp:
            norm_ecmp_mlu = optimal_mlu / ecmp_mlu
            line += str(norm_ecmp_mlu) + ', ' + str(ecmp_mlu) + ', '

        if eval_delay:
            solution = self.optimal_routing_delay(tm_idx)
            optimal_delay = self.eval_optimal_routing_delay(tm_idx, solution) 

            line += str(optimal_delay/delay) + ', ' 
            line += str(optimal_delay/crit_delay) + ', ' 
            line += str(optimal_delay/topk_delay) + ', ' 
            line += str(optimal_delay/optimal_mlu_delay) + ', '
            if ecmp:
                line += str(optimal_delay/ecmp_delay) + ', '
        
            assert tm_idx in self.load_multiplier, (tm_idx)
            line += str(self.load_multiplier[tm_idx]) + ', '

        print(line[:-2])
        
    
        
        
        
    def evaluate3(self,tm_idx):
        mlu_current,current_solutions = self.current_mlu_optimal_routing_mlu(tm_idx)
        mlu_next,next_solutions = self.current_mlu_optimal_routing_mlu(tm_idx+1)
        mlu_next_using_current,_  = self.eval_optimal_routing_mlu(tm_idx+1, current_solutions, eval_delay=False)
        return mlu_current,mlu_next,mlu_next_using_current,current_solutions,next_solutions
    def evaluate2(self, env,config,tm_idx,all_tms,commitment_window, look_ahead_window,topology_file,scheme,max_move,actions=None, ecmp=True, eval_delay=False):
        
#         toplogy_t_solution_mlu_result = config.testing_results
#         time_index = tm_idx
#         each_flow_shortest_path = env.topology.get_each_flow_shortest_paths()
#         each_flow_paths,each_path_id,each_id_path = self.get_each_flow_paths_and_path_id(topology_file)
#         each_flow_edges_hindsight_approach = self.get_hindsight_set_of_paths(tm_idx,
#                                                                              look_ahead_window,each_flow_paths,
#                                                                              each_path_id,each_id_path,
#                                                                              each_flow_shortest_path,topology_file,config)
        
#         hindsight_mlus,hindsight_solutions = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
#       each_flow_edges_hindsight_approach)
        #print("this is the scheme ",scheme)
        time_index = tm_idx
        
        #each_flow_shortest_path = env.topology.get_each_flow_shortest_paths()
        #each_flow_paths,each_path_id,each_id_path = self.get_each_flow_paths_and_path_id(env,config,topology_file)
        each_flow_oblivious_paths = env.topology.get_each_flow_oblivious_paths(config.raeke_paths)
        if scheme =="ECMP":
            """for ECMP mlu"""
            solutions = {}
            ecmp_mlus = self.eval_ecmp_traffic_distribution2(tm_idx,look_ahead_window, eval_delay=eval_delay)

            
            return ecmp_mlus,solutions,1,solutions
        elif scheme =="Shortest":
            
            each_flow_shortest_path_edges = {}
            for flow,path in self.each_flow_shortest_path.items():
                #print("this is the path for shortest path ")
                #print(path)
                #import pdb
                for node_indx in range(len(path)-1):
                    try:
                        if (path[node_indx],path[node_indx+1]) not in each_flow_shortest_path_edges[flow]:
                            each_flow_shortest_path_edges[flow].append((path[node_indx],path[node_indx+1]))
                            each_flow_shortest_path_edges[flow].append((path[node_indx+1],path[node_indx]))
                    except:
                        each_flow_shortest_path_edges[flow]=[(path[node_indx],path[node_indx+1])]
                        each_flow_shortest_path_edges[flow].append((path[node_indx+1],path[node_indx]))
            shortest_path_mlus,shortest_path_solutions  = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
                                                                   each_flow_shortest_path_edges)
            shortest_path_mlus2,shortest_path_solutions2  = self.mlu_routing_selected_paths_fixed_rates(tm_idx,look_ahead_window,
                                                                   each_flow_shortest_path_edges)
            solution ={}
            return shortest_path_mlus,shortest_path_solutions,shortest_path_mlus2,shortest_path_solutions2
        elif scheme =="Oblivious":
            """for Oblivious routing"""
            
            each_flow_oblivious_paths_edges = env.topology.get_oblivious_paths_each_flow_edges(config.raeke_paths,
                                                                                          max_move,self.each_flow_shortest_path)
#             oblivious_mlus,oblivious_solutions = self.mlu_routing_selected_paths_oblivious(tm_idx,look_ahead_window,
#                                                                              each_flow_oblivious_paths_edges)
            oblivious_mlus,oblivious_solutions  = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
                                                                   each_flow_oblivious_paths_edges)
            solution = {}
            return oblivious_mlus,oblivious_solutions,1,solution
        elif scheme =="Oblivious2":
            """for Oblivious routing"""
            
            each_flow_oblivious_paths_edges = env.topology.get_oblivious_paths_each_flow_edges(config.raeke_paths,
                                                                                          max_move,self.each_flow_shortest_path)
            oblivious_mlus,oblivious_solutions = self.mlu_routing_selected_paths_oblivious2(tm_idx,look_ahead_window,
                                                                             each_flow_oblivious_paths_edges)
            
            oblivious_mlus,oblivious_solutions  = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
                                                                   each_flow_oblivious_paths_edges)
            solution = {}
            return oblivious_mlus,oblivious_solutions,1,solution
        elif scheme =="MLU-greedy":
            """for MLU_greedy"""
            time_index = tm_idx
            mlu_greedy_mlus,mlu_greedy_solutions = self.mlu_optimal_routing_mlu(tm_idx,look_ahead_window)
            #print("we have scheme %s and result is %s"%(scheme,sum(mlu_greedy_mlus)/len(mlu_greedy_mlus)))
            solution = {}
            return mlu_greedy_mlus,mlu_greedy_solutions,1,solution
        elif scheme=="Optimal":
#             each_flow_all_paths,each_path_edges = self.get_set_of_all_paths(tm_idx,
#                                                                                  look_ahead_window,each_flow_paths,
#                                                                                  each_path_id,each_id_path,
#                                                                                  each_flow_shortest_path,each_flow_oblivious_paths,
#                                                                                  topology_file,config)
            
            
            
           
 
           
            import pdb
            #pdb.set_trace()
            optimal1_mlus,optimal1_solutions,optimal2_mlus,optimal2_solutions = self.optimal_solution_for_robust_paths(
                                                                            tm_idx,look_ahead_window)
        
            return optimal1_mlus,optimal1_solutions,optimal2_mlus,optimal2_solutions
        elif scheme =="DRL":
            """for RL approach"""
            time_index = tm_idx
            max_move = len(actions)
            each_flow_edges_rl_approach = self.get_suggested_paths_by_rl(actions)
            
#             each_flow_oblivious_paths_edges = env.topology.get_oblivious_paths_each_flow_edges(config.raeke_paths,
#                                                                                           max_move,self.each_flow_shortest_path)
            
#             oblivious_max_node_number = []
#             rl_max_node_number = []
#             for flow,edges in each_flow_edges_rl_approach.items():
#                 for edge in edges:
#                     rl_max_node_number.append(edge[0])
#                     rl_max_node_number.append(edge[1])
#                 print("for flow %s we have edges in RL        %s "%(flow,edges))
#                 print("for flow %s we have edges in oblivious %s"%(flow,each_flow_oblivious_paths_edges[flow]))
#                 for edge in each_flow_oblivious_paths_edges[flow]:
#                     oblivious_max_node_number.append(edge[0])
#                     oblivious_max_node_number.append(edge[1])
#                 if (len(each_flow_oblivious_paths_edges[flow])%2!=0):
#                     print("error")
#             print("max rl %s max  oblivios %s"%(max(rl_max_node_number),max(oblivious_max_node_number)))
#             print("min rl %s min  oblivios %s"%(min(rl_max_node_number),min(oblivious_max_node_number)))
            #print("unique rl %s unique  oblivios %s"%(len(list(set(rl_max_node_number))),len(list(set(oblivious_max_node_number)))))
            actions.sort()
#             for path in actions:
#                 print("for DRL sceme we have %s path %s"%(len(actions),path))
            solution = {}
            rl_mlus,rl_solutions = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
                                                                   each_flow_edges_rl_approach)
#             print("MLU using rl is ",sum(rl_mlus)/len(rl_mlus))
#             for limit in [1,2,3,4,5,6,7,8,9,10,20,40,80,120,200,300,400,1000]:
#                 backup_each_flow_edges_rl_approach = {}
#                 for flow ,edges in each_flow_edges_rl_approach.items():
#                     backup_each_flow_edges_rl_approach[flow] = edges
#                 counter = 0
#                 for edge in self.lp_links:
#                     print("this is an edge ",edge)
#                 for flow ,edges in each_flow_oblivious_paths_edges.items():
#                     if flow != (0,3):
#                         if counter <limit:
#                             print("for flow ",flow)
#                             #for edge in edges:
#                             #    if edge not in each_flow_edges_rl_approach[flow]:
#                             print("we replace this edges ",backup_each_flow_edges_rl_approach[flow])
#                             print("with these ",edges)
#                             backup_each_flow_edges_rl_approach[flow] = edges
#                             for edge in edges:
#                                 if edge not in self.lp_links:
#                                     print("this is an ERROR!! link  does not exist in lp_links",edge)
#                             counter+=1
#                 print("done adding edge to the default rl edges",limit)
#                 oblivious_mlus,oblivious_solutions  = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
#                                                                        backup_each_flow_edges_rl_approach)

#                 print("the mlu using oblivious is ",len(oblivious_mlus)/len(oblivious_mlus))
#                 time.sleep(5)
            return rl_mlus,rl_solutions,1,solution
    
        
        
                #print("oblivious edge flows",len(list(each_flow_oblivious_paths_edges.keys())))
        #print("shortest_paths",len(list(each_flow_shortest_path.keys())))
#         for flow,edges in each_flow_oblivious_paths_edges.items():
#             print("flow %s has edges %s"%(flow,edges))
        

    
        
        
        
        
        
                
        
#         each_flow_all_paths,each_path_edges = self.get_set_of_all_paths(tm_idx,
#                                                                              look_ahead_window,each_flow_paths,
#                                                                              each_path_id,each_id_path,
#                                                                              each_flow_shortest_path,each_flow_oblivious_paths,topology_file,config)
        
#         hindsight_mlus,hindsight_solutions = self.optimal_solution_for_robust_paths(tm_idx,look_ahead_window,
#                                                                              each_flow_all_paths,each_path_edges)
        
        
        
        
        
        
#         if sum(oblivious_mlus)/len(oblivious_mlus) < sum(hindsight_mlus)/len(hindsight_mlus):
#             hindsight_mlus,hindsight_solutions = self.mlu_routing_selected_paths(tm_idx,look_ahead_window,
#                                                                              each_flow_oblivious_paths_edges)
        
        
        

        
        
        
      
       
        

        
                
                
                

