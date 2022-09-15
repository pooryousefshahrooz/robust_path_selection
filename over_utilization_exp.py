#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import os
import ast
import numpy as np
from absl import app
from absl import flags

import tensorflow as tf
from env import Environment
from game import CFRRL_Game
from model import Network
from config import get_config
import csv
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')
def get_paths_info(topology_name,each_topology_each_t_each_f_paths):
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
                    all_the_paths.add(p)

    paths_number = []
    for time, paths in each_t_paths.items():
        paths_number.append(len(paths))
    avg_paths_per_time = int((sum(paths_number)/len(paths_number)))
     
    return len(list(all_the_paths)),avg_paths_per_time

def get_other_schemes_mlu(scheme):
    
    return mlu_greedy_solutions,
def sim(config, network,tm_idx,env, game,commitment_window,look_ahead_window,scheme):
    counter = 0
#        print("tm_idx is " ,tm_idx)
    counter+=1
    #print("done %s from %s traffic matrices for commitment %s lookahead %s"%(counter,len(game.tm_indexes),commitment_window,look_ahead_window))
    state = game.get_state(tm_idx)
    if config.method == 'actor_critic':
        policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
    elif config.method == 'pure_policy':
        policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
    actions = policy.argsort()[-game.max_moves:]
    max_move = game.max_moves
    #game.evaluate3(env,config,tm_idx,len(game.tm_indexes),training_epoch,commitment_window,look_ahead_window,config.topology_file, actions,training_epoch, eval_delay=False) 
    drl_mlus,drl_solutions,_,_ = game.evaluate2(env,config,tm_idx,len(game.tm_indexes),commitment_window,look_ahead_window,config.topology_file,scheme, max_move,actions, eval_delay=False) 
    solutions2 = {}
    return drl_mlus,drl_solutions,1,solutions2
def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env,2,2,400,400)
    game.max_moves = 400
    max_value = 400

    all_tms = len(game.tm_indexes)

    
    #network = Network(config, game.state_dims, game.action_dim, game.max_moves,commitment_window,look_ahead_window,training_epoch)
    #print("This is the chkpoint path ",network.ckpt_dir)
    #step = network.restore_ckpt(FLAGS.ckpt)


    for tm_idx in range(0,len(game.tm_indexes)-2):
        
        mlu_current,mlu_next,mlu_next_using_current,current_solutions,next_solutions = game.evaluate3(tm_idx)
        
        toplogy_t_solution_mlu_result = config.over_utilization_results_file
        
        with open(toplogy_t_solution_mlu_result, 'a') as newFile:                                
            newFileWriter = csv.writer(newFile)
            for item,v in current_solutions.items():
                #print('flow index %s from node %s to the next node %s ratio %s'%(item[0],item[1],item[2],v))
                newFileWriter.writerow([config.topology_file,tm_idx,
                mlu_current,mlu_next,mlu_next_using_current
                ,item[0],item[1],item[2],v,
                next_solutions[(item[0],item[1],item[2])]
                ])
        print("done %s from %s with current mlu %s next mlu %s over %s "%(tm_idx,len(game.tm_indexes)-2,mlu_current,mlu_next,mlu_next_using_current))
                
if __name__ == '__main__':
    app.run(main)


# In[ ]:





# In[ ]:




