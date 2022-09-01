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
    
    each_flow_shortest_paths = env.topology.get_each_flow_shortest_paths()
    path_counter,avg_paths_per_time = get_paths_info(config.topology_file,config.each_topology_each_t_each_f_paths)
    #for commitment_window in range(2,int(config.commitment_window_range)):
    for commitment_window in [4,6,2,8,10]:
        #for look_ahead_window in range(6,int(config.look_ahead_window_range)):
        for look_ahead_window in [2,4,6,8,10]:
            """we first find the candidate paths and use it for action dimention"""
            game = CFRRL_Game(config, env,commitment_window,look_ahead_window,path_counter,avg_paths_per_time)
            game.set_paths_info(env,config.topology_file,config.each_topology_each_t_each_f_paths)
            game.max_moves = avg_paths_per_time
            max_value = avg_paths_per_time
            training_epochs = game.get_all_trainig_epochs(commitment_window,look_ahead_window)
            #training_epochs = [1,9,19,29,39,49]
            actions = []
            all_tms = len(game.tm_indexes)
            if not config.training_epochs_experiment:
                training_epochs = [max(training_epochs)]
            for training_epoch in training_epochs:
                network = Network(config, game.state_dims, game.action_dim, game.max_moves,commitment_window,look_ahead_window,training_epoch)
                #print("This is the chkpoint path ",network.ckpt_dir)
                step = network.restore_ckpt(FLAGS.ckpt)
                if config.method == 'actor_critic':
                    learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
                elif config.method == 'pure_policy':
                    learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
                print('\nstep %d, learning rate: %f\n'% (step, learning_rate))
                    
                for tm_idx in range(commitment_window,len(game.tm_indexes)-look_ahead_window-1):
                    if training_epoch==min(training_epochs):
                        mlu_greedy_mlus,mlu_greedy_solutions,_,_ = game.evaluate2(env,config,tm_idx,len(game.tm_indexes),commitment_window,look_ahead_window,config.topology_file,"MLU-greedy", max_value,actions, eval_delay=False)
                        ecmp_mlus,_,_,_ = game.evaluate2(env,config,tm_idx,len(game.tm_indexes),commitment_window,look_ahead_window,config.topology_file, "ECMP",max_value,actions, eval_delay=False)
                        oblivious_mlus,_,_,_ = game.evaluate2(env,config,tm_idx,len(game.tm_indexes),commitment_window,look_ahead_window,config.topology_file,"Oblivious", max_value,actions, eval_delay=False)
                        oblivious2_mlus,_,_,_ = game.evaluate2(env,config,tm_idx,len(game.tm_indexes),commitment_window,look_ahead_window,config.topology_file,"Oblivious2", max_value,actions, eval_delay=False)
                        optimal1_mlus,_,optimal2_mlus,_ = game.evaluate2(env,config,tm_idx,len(game.tm_indexes),commitment_window,look_ahead_window,config.topology_file,"Optimal", max_value,actions, eval_delay=False)
                    toplogy_t_solution_mlu_result = config.testing_results
                    #print("training_epochs",training_epochs)
                
                    """we modify and set the max move to the avg used path at each t for  all flows"""
                    
                    rl_mlus,_,_,_= sim(config, network,tm_idx,env, game,commitment_window,look_ahead_window,"DRL")
                    mlu_index = 0
                    time_index = tm_idx
                    mlu_index = 0 
                    for sol in mlu_greedy_solutions:
                        print("c:",commitment_window,"w:",look_ahead_window,
                              "epoch:",training_epoch,"t:",time_index,
                              "all_tms:",all_tms,
                                    "optimal",round(optimal1_mlus[mlu_index],3),
                                    "mlu_greedy",
                                    round(mlu_greedy_mlus[mlu_index],3),
                                    "rl",
                                    round(rl_mlus[mlu_index],3),
                                    "ecmp",round(ecmp_mlus[mlu_index],3),
                                    "oblivious",round(oblivious_mlus[mlu_index],3),
                                      "oblivious2",round(oblivious2_mlus[mlu_index],3))
                        if round(oblivious_mlus[mlu_index],3) <round(mlu_greedy_mlus[mlu_index],3):
                            print("this does not make sense!!!!!!!")
                        mlu_index+=1
                    mlu_index = 0
                    with open(toplogy_t_solution_mlu_result, 'a') as newFile:                                
                        newFileWriter = csv.writer(newFile)
                        solution_indx = 0
                        for solution in mlu_greedy_solutions:
                            for item,v in solution.items():
                                #print('flow index %s from node %s to the next node %s ratio %s'%(item[0],item[1],item[2],v))
                                newFileWriter.writerow([config.topology_file,commitment_window,
                                look_ahead_window,time_index,
                                "optimal",round(optimal1_mlus[mlu_index],3),
                                "mlu_greedy",item[0],item[1],item[2],
                                mlu_greedy_solutions[solution_indx][(item[0],item[1],item[2])],
                                round(mlu_greedy_mlus[mlu_index],3),
                                "rl",
                                round(rl_mlus[mlu_index],3),
                                "ecmp",round(ecmp_mlus[mlu_index],3),
                                "oblivious",round(oblivious_mlus[mlu_index],3),
                                "oblivious2",round(oblivious2_mlus[mlu_index],3),training_epoch])
                            time_index+=1
                            mlu_index+=1
                            solution_indx+=1
                
                
                
                
                
                
if __name__ == '__main__':
    app.run(main)


# In[ ]:





# In[ ]:




