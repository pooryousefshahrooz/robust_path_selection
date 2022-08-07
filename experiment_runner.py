#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import ast
# from config import get_config
from train import train_main


import tensorflow as tf    
from env import Environment
from game import CFRRL_Game
from model import Network
from config import get_config
import os
import numpy as np
# from absl import flags

# from config import get_config

# FLAGS = flags.FLAGS


# In[ ]:





def extract_paths(topology_name):
    """in this function, we get the B=4 most used paths by each flow
    
    we get all the used path by each flow over all times and get top 4 of them"""
    path_counter = 0
    each_path_id = {}
    set_of_times = set([])
    each_flow_path_usage = {}
    print('topology is ',topology_name)
    with open('each_topology_each_t_each_f_paths.txt') as file:
        lines = file.readlines()
        for line in lines:
            if topology_name in line:#ATT_topology_file_modified:189:1->20: ['1', '13', '9', '4', '20'],['1', '9', '4', '20']
                time = line.split(":")[1]
                flow = line.split(":")[2]
                set_of_times.add(time)
                paths = line.split(flow+":")[1]
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
                    #print('this is a path ',p)
                    try:
                        each_flow_path_usage[flow][p]+=1
                    except:
                        try:
                            each_flow_path_usage[flow][p]= 1
                        except:
                            each_flow_path_usage[flow]= {}
                            each_flow_path_usage[flow][p]= 1
                
                
    each_flow_top_four_used_paths = {}
    for each_flow,path_usage in each_flow_path_usage.items():
        top_used_numbers = []
        for p, usage in path_usage.items():
            top_used_numbers.append(usage)
        top_used_numbers.sort()
        if len(top_used_numbers) >3:
            top_used_numbers = top_used_numbers[-4:]
        else:
            if len(top_used_numbers)==3:
                top_used_numbers.append(top_used_numbers[2])
            elif len(top_used_numbers)==2:
                top_used_numbers.append(top_used_numbers[1])
                top_used_numbers.append(top_used_numbers[1])
            elif len(top_used_numbers)==1:
                top_used_numbers.append(top_used_numbers[0])
                top_used_numbers.append(top_used_numbers[0])
                top_used_numbers.append(top_used_numbers[0])
            top_used_numbers = top_used_numbers
        each_flow_top_four_used_paths[each_flow] = top_used_numbers
        
    each_flow_top_used_paths = {}
    for each_flow,top_used_numbers in each_flow_top_four_used_paths.items():
        for flow,paths in each_flow_path_usage.items():
            if each_flow ==flow:
                for p, usage_times in paths.items():
                    if usage_times in top_used_numbers:
                        try:
                            if p not in each_flow_top_used_paths[flow]:
                                each_flow_top_used_paths[flow].append(p)
                        except:
                            each_flow_top_used_paths[flow]= [p]
    import pdb
#     for flow, top_paths in each_flow_top_used_paths.items():
#         print("flow %s paths %s"%(flow,top_paths))
        
    each_flow_most_used_paths = {}
    for flow, paths in each_flow_top_used_paths.items():
        new_paths = []
        for p in paths:
            new_paths.append(p)
        if len(new_paths)==3:
            new_paths.append(p)
        elif len(new_paths)==2:
            new_paths.append(p)
            new_paths.append(p)
        elif len(new_paths)==1:
            new_paths.append(p)
            new_paths.append(p)
            new_paths.append(p)
        each_flow_most_used_paths[flow] = new_paths
    for flow, top_paths in each_flow_most_used_paths.items():
        path_counter+=4
        #print("flow %s paths %s"%(flow,len(top_paths)))
    return path_counter,each_flow_most_used_paths
    #pdb.set_trace()
    #return path_counter,each_path_id
    
#config = get_config(FLAGS)
#print('we got the config')
path_counter,each_path_id= extract_paths('Abilene')
commitments = 8
look_ahead_windows = 10
for commit in range(2,int(commitments)):
    for lookahead in range(2,int(look_ahead_windows)):
        #config.traffic_file = 'TM'

        train_main()
        #config.traffic_file = 'TM2'

        #test(config)

        #app.run(test.main)


# In[ ]:





# In[ ]:



# stringA = "['21','42', '15'],['43','55','32']"
# stringA = "['0', '1']"
# # Given string
# print("Given string", stringA)
# print(type(stringA))
# # String to list
# res = ast.literal_eval(stringA)
# # Result and its type
# print("final list", res)
# print(type(res))
# print(len(res))
# for item in res:
#     print(item)

# import ast
# stringA = "[21,42, 15]"
# # Given string
# print("Given string", stringA)
# print(type(stringA))
# # String to list
# res = ast.literal_eval(stringA)
# # Result and its type
# print("final list", res)
# print(type(res))


# In[ ]:




