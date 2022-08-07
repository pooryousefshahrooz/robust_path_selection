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

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')
def get_paths_number(topology_name,each_topology_each_t_each_f_paths):
    
    
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
                    
    paths_number = []
    for time, paths in each_t_paths.items():
        #print('for time %s we had %s paths'%(time,len(paths)))
        paths_number.append(len(paths))
    avg_paths_per_time = int((sum(paths_number)/len(paths_number)))
    return len(list(all_the_paths)),avg_paths_per_time
def sim(config, network,env, game,commitment_window,look_ahead_window,training_epoch):
    counter = 0
    for tm_idx in range(commitment_window,len(game.tm_indexes)-look_ahead_window-1):
#        print("tm_idx is " ,tm_idx)
        counter+=1
        #print("done %s from %s traffic matrices for commitment %s lookahead %s"%(counter,len(game.tm_indexes),commitment_window,look_ahead_window))
        state = game.get_state(tm_idx)
        if config.method == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.method == 'pure_policy':
            policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-game.max_moves:]

        game.evaluate2(env,config,tm_idx,len(game.tm_indexes),training_epoch,commitment_window,look_ahead_window,config.topology_file, actions,training_epoch, eval_delay=False) 

def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    
    each_flow_shortest_paths = env.topology.get_each_flow_shortest_paths()
    path_counter,avg_paths_per_time = get_paths_number(config.topology_file,config.each_topology_each_t_each_f_paths)
    for commitment_window in range(2,int(config.commitment_window_range)):
        for look_ahead_window in range(2,int(config.look_ahead_window_range)):
            """we first find the candidate paths and use it for action dimention"""
            game = CFRRL_Game(config, env,commitment_window,look_ahead_window,path_counter,avg_paths_per_time)
            training_epochs = game.get_all_trainig_epochs(commitment_window,look_ahead_window)
            #training_epochs = [1,9,19,29,39,49]
            
            if not config.training_epochs_experiment:
                training_epochs = [max(training_epochs)]
            print("training_epochs",training_epochs)
            for training_epoch in training_epochs:
                
                """we modify and set the max move to the avg used path at each t for  all flows"""
                game.max_moves = avg_paths_per_time

                network = Network(config, game.state_dims, game.action_dim, game.max_moves,commitment_window,look_ahead_window,training_epoch)
                print("This is the chkpoint path ",network.ckpt_dir)
                step = network.restore_ckpt(FLAGS.ckpt)
                if config.method == 'actor_critic':
                    learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
                elif config.method == 'pure_policy':
                    learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
                print('\nstep %d, learning rate: %f\n'% (step, learning_rate))

                sim(config, network,env, game,commitment_window,look_ahead_window,training_epoch)
if __name__ == '__main__':
    app.run(main)

