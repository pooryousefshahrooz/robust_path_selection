#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from absl import app
from absl import flags
import ast
import tensorflow as tf
from env import Environment
from game import CFRRL_Game
from model import Network
from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_agents',1, 'number of agents')
flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseline')
flags.DEFINE_integer('num_iter', 20, 'Number of iterations each agent would run')
# print(FLAGS.num_agents)
import pdb
# pdb.set_trace()
GRADIENTS_CHECK=False


# In[ ]:


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


# In[ ]:


def central_agent(config, game,commitment_window,look_ahead_window, model_weights_queues, experience_queues):
    network = Network(config, game.state_dims, game.action_dim, game.max_moves,commitment_window,look_ahead_window,1, master=True)
    print("config.max_step",config.max_step)
    import pdb
    #pdb.set_trace()
    network.save_hyperparams(config)
    start_step = network.restore_ckpt()
    number_of_training_epochs = 0
    for step in tqdm(range(start_step, config.max_step), ncols=70, initial=start_step):
        network.ckpt.step.assign_add(1)
        model_weights = network.model.get_weights()
        number_of_training_epochs+=1
#         print("***********************")
#         print(len(model_weights_queues))
#         print(FLAGS.num_agents)
#         print("************.....**********")
#         import pdb
        #pdb.set_tracce()
        for i in range(FLAGS.num_agents):
            model_weights_queues[i].put(model_weights)

        if config.method == 'actor_critic':
            #assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []

            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent = experience_queues[i].get()
              
                assert len(s_batch_agent) == FLAGS.num_iter,                     (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent
           
            assert len(s_batch)*game.max_moves == len(a_batch)
            #used shared RMSProp, i.e., shared g
            actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
            #print('this is game.action_dim',game.action_dim)
            #print('this is a_batch',a_batch)
#            print('this is actions',actions)
            
#             for item in actions:
#                 print('len of each item in actions',len(item),item)
            import pdb
            #print('len of a_batch',len(a_batch),'len of actions',len(actions))
            #print('this is the item in array',np.array(a_batch))
            
            #print(len(np.eye(game.action_dim, dtype=np.float32)),len(np.array(a_batch)))
            #print("****")
#             print(a_batch)
#             print(len(a_batch))
#             print(actions)
#             print(len(actions))
#             print(len(np.array(a_batch)))
#             print(len(np.array([0,1,2])))
#             print("***********")
#             print('number of actions',len(actions))
            #pdb.set_trace()
            
            value_loss, entropy, actor_gradients, critic_gradients = network.actor_critic_train(np.array(s_batch), 
                                                                    actions, 
                                                                    np.array(r_batch).astype(np.float32), 
                                                                    config.entropy_weight)
       
            if GRADIENTS_CHECK:
                for g in range(len(actor_gradients)):
                    assert np.any(np.isnan(actor_gradients[g])) == False, ('actor_gradients', s_batch, a_batch, r_batch, entropy)
                for g in range(len(critic_gradients)):
                    assert np.any(np.isnan(critic_gradients[g])) == False, ('critic_gradients', s_batch, a_batch, r_batch)

            if step % config.save_step == config.save_step - 1:
                print("check point dir befor update out is ",network.ckpt_dir)
                network.update_chkpt_saving_dir(config,commitment_window,look_ahead_window,step)
                print("check point dir now out is ",network.ckpt_dir)
                network.save_ckpt(_print=True)
                
                #log training information
                actor_learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
                avg_value_loss = np.mean(value_loss)
                avg_reward = np.mean(r_batch)
                avg_entropy = np.mean(entropy)
            
                network.inject_summaries({
                    'learning rate': actor_learning_rate,
                    'value loss': avg_value_loss,
                    'avg reward': avg_reward,
                    'avg entropy': avg_entropy
                    }, step)
                print('lr:%f, value loss:%f, avg reward:%f, avg entropy:%f'%(actor_learning_rate, avg_value_loss, avg_reward, avg_entropy))

        elif config.method == 'pure_policy':
            #assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []
            ad_batch = []

            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent, ad_batch_agent = experience_queues[i].get()
              
                assert len(s_batch_agent) == FLAGS.num_iter,                     (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent), len(ad_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent
                ad_batch += ad_batch_agent
           
            assert len(s_batch)*game.max_moves == len(a_batch)
            #used shared RMSProp, i.e., shared g
            actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
            entropy, gradients = network.policy_train(np.array(s_batch), 
                                                      actions, 
                                                      np.vstack(ad_batch).astype(np.float32), 
                                                      config.entropy_weight)

            if GRADIENTS_CHECK:
                for g in range(len(gradients)):
                    assert np.any(np.isnan(gradients[g])) == False, (s_batch, a_batch, r_batch)
            
            if step % config.save_step == config.save_step - 1:
                network.update_chkpt_saving_dir(config,commitment_window,look_ahead_window,step)
                network.save_ckpt(_print=True)
                
                #log training information
                learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
                avg_reward = np.mean(r_batch)
                avg_advantage = np.mean(ad_batch)
                avg_entropy = np.mean(entropy)
                network.inject_summaries({
                    'learning rate': learning_rate,
                    'avg reward': avg_reward,
                    'avg advantage': avg_advantage,
                    'avg entropy': avg_entropy
                    }, step)
                print('lr:%f, avg reward:%f, avg advantage:%f, avg entropy:%f'%(learning_rate, avg_reward, avg_advantage, avg_entropy))

def agent(agent_id, config, game,commitment_window,look_ahead_window,tm_subset, model_weights_queue, experience_queue):
    random_state = np.random.RandomState(seed=agent_id)
    network = Network(config, game.state_dims, game.action_dim, game.max_moves,commitment_window,look_ahead_window, 1, master=False)

    # initial synchronization of the model weights from the coordinator 
    model_weights = model_weights_queue.get()
    network.model.set_weights(model_weights)

    idx = 0
    s_batch = []
    a_batch = []
    r_batch = []
    if config.method == 'pure_policy':
        ad_batch = []
    run_iteration_idx = 0
    num_tms = len(tm_subset)
    #print('this is random state before shuffling',random_state)
    #random_state.shuffle(tm_subset)
    #print('this is random state after shuffling',random_state)
    run_iterations = FLAGS.num_iter
    #print("****************--------------------------------------***************run_iterations",run_iterations)
    import pdb
    #pdb.set_trace()
    while True:
        
        tm_idx = tm_subset[idx]
        #print("idx is %s and tm_idx is %s and num_tms is %s"%(idx,tm_idx,num_tms))
        #state
        if tm_idx <=num_tms -(look_ahead_window+2):
            state = game.get_state(tm_idx)
            #print(state)
            import pdb
            #pdb.set_trace()
            s_batch.append(state)
            #action
            if config.method == 'actor_critic':    
                policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
                #print("\n \n \n")
                #print('here is the policy',policy)
                import pdb
                #pdb.set_trace()
            elif config.method == 'pure_policy':
                policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
            assert np.count_nonzero(policy) >= game.max_moves, (policy, state)
            actions = random_state.choice(game.action_dim, game.max_moves, p=policy, replace=False)
            #print(len(policy))

            #print('and here is the actions',game.action_dim,game.max_moves,actions)

            #print("these must be the same as actions",np.argpartition(policy, -4)[-4:])
            import pdb
            #pdb.set_trace()
            for a in actions:
                a_batch.append(a)

            #reward
            #reward = game.reward(tm_idx, actions)
            """ safe online learning section"""

            """we check if for all the flows there is atleast one path in the chosen paths
            if not, we will use the shortest path for that flow"""

            #each_flow_shortest_path = env.get_each_flow_shortest_paths()
#             for flow, shortest_path in each_flow_shortest_path.items():
#                 print("for flow %s we have shortest path %s "%(flow,shortest_path))
            import pdb

            #print("the rl has selected %s paths "%(len(actions)))
            for flow,paths in game.each_flow_paths.items():
                covered_flow = False
                for path in paths:
                    
                    if not covered_flow:
                        if path in actions:
                            covered_flow =True 
                if not covered_flow:
#                     print("flow  did not have any candidate path in chosen paths",flow)
                    flow_shortest_path = game.each_flow_shortest_path[flow]
#                     print(flow_shortest_path)
                    #print(each_path_id)
                    path_id = game.each_path_id[tuple(flow_shortest_path)]
            
                    actions = np.append(actions, path_id)
#                     print("we added path id %s for safety for flow %s"%(path_id,flow))

            """end of safe online learning section"""

    #         for path_id in actions:
    #             print("we have chosen and safed action %s %s"%(path_id,len(actions)))

            each_flow_selected_paths = {}
            """we now add the edges for each flow from the selected actions"""
            each_flow_edges = {}
            for flow in game.each_flow_shortest_path:
                for path_id in actions:
                    path = game.each_path_edges[path_id]
                    if path_id in game.each_flow_paths[flow]:
                        this_path_edges = game.each_path_edges[path_id]
                        try:
                            each_flow_selected_paths[flow].append(path_id)
                        except:
                            each_flow_selected_paths[flow]=[path_id]
                        for edge in path:
#                             print("we are adding edge ",edge)
                            try:
                                if edge not in each_flow_edges[flow] and (edge[1],edge[0]) not in each_flow_edges[flow]:
                                    each_flow_edges[flow].append(edge)
                                    each_flow_edges[flow].append((edge[1],edge[0]))
                            except:
                                each_flow_edges[flow]=[edge]
                                each_flow_edges[flow].append((edge[1],edge[0]))
                                
                                
#                 if flow ==(0, 5):
#                     print("each_flow_paths",each_flow_paths)
#                     print("here are the edges for this flow ",each_flow_edges[flow])

#             for flow,edges in each_flow_edges.items():
#                 print("this flow %s has these edges %s"%(flow,edges))
            p_counter = 0
            for flow,paths in each_flow_selected_paths.items():
                for path in paths:
                    p_counter+=1
                    #print("flow %s uses path %s "%(flow,path))
            #print("we have chosen %s paths for % flows "%(p_counter,len(list(each_flow_selected_paths.keys()))))
            #pdb.set_trace()

    #         for path_id in actions:
    #             each_flow_edges[]
            reward = game.reward2(tm_idx,num_tms,look_ahead_window,each_flow_edges,config.topology_file,config.printing_flag)
            r_batch.append(reward)

            if config.method == 'pure_policy':
                #advantage
                if config.baseline == 'avg':
                    ad_batch.append(game.advantage(tm_idx, reward))
                    game.update_baseline(tm_idx, reward)
                elif config.baseline == 'best':
                    best_actions = policy.argsort()[-game.max_moves:]
                    best_reward = game.reward(tm_idx, best_actions)
                    ad_batch.append(reward - best_reward)

            run_iteration_idx += 1
            if run_iteration_idx >= run_iterations:
                # Report experience to the coordinator                          
                if config.method == 'actor_critic':    
                    experience_queue.put([s_batch, a_batch, r_batch])
                elif config.method == 'pure_policy':
                    experience_queue.put([s_batch, a_batch, r_batch, ad_batch])

                #print('report', agent_id)

                # synchronize the network parameters from the coordinator
                model_weights = model_weights_queue.get()
                network.model.set_weights(model_weights)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                if config.method == 'pure_policy':
                    del ad_batch[:]
                run_iteration_idx = 0

        # Update idx
        idx += 1
        if idx+game.look_ahead_window+2 >= num_tms:
           random_state.shuffle(tm_subset)
           idx = 0


# In[ ]:


def main(_):
    #cpu only
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')
    #tf.debugging.set_log_device_placement(True)

    config = get_config(FLAGS) or FLAGS
    
    """we get the number of possible paths in the network, 
    avg paths used by all flows over times and each flows all used paths"""
    #path_counter,avg_paths_per_time,each_flow_paths= extract_paths(config.topology_file,config.each_topology_each_t_each_f_paths)
    path_counter,avg_paths_per_time = get_paths_info(config.topology_file,config.each_topology_each_t_each_f_paths)
#     each_path_id = {}
#     id_counter = 0
#     each_id_path = {}
#     covered_paths = []
#     each_path_edges = {}
#     new_each_flow_paths = {}
#     for flow,paths in each_flow_paths.items():
#         for p in paths:
#             covered_paths.append(p)
#             each_path_id[tuple(p)] = id_counter
#             try:
#                 new_each_flow_paths[flow].append(id_counter)
#             except:
#                 new_each_flow_paths[flow] = [id_counter]
#             each_id_path[id_counter] = tuple(p)
#             path = tuple(p)
#             for node_indx in range(len(path)-1):
#                 try:
#                     if (path[node_indx],path[node_indx+1]) not in each_path_edges[id_counter]:
#                         each_path_edges[id_counter].append((path[node_indx],path[node_indx+1]))
#                         each_path_edges[id_counter].append((path[node_indx+1],path[node_indx]))
#                 except:
#                     each_path_edges[id_counter]=[(path[node_indx],path[node_indx+1])]
#                     each_path_edges[id_counter].append((path[node_indx+1],path[node_indx]))
            
#             id_counter+=1
            
    env = Environment(config,is_training=True)
    #print("env.num_nodes",env.num_nodes)
    import pdb
    #pdb.set_trace()
#     each_flow_shortest_paths = env.topology.get_each_flow_shortest_paths()
#     for flow,p in each_flow_shortest_paths.items():
#         p = tuple(p)
#         if p not in covered_paths:
#             each_path_id[tuple(p)] = id_counter
#             try:
#                 new_each_flow_paths[flow].append(id_counter)
#             except:
#                 new_each_flow_paths[flow] = [id_counter]
                
#             each_id_path[id_counter] = tuple(p)
#             path = tuple(p)
#             for node_indx in range(len(path)-1):
#                 try:
#                     if (path[node_indx],path[node_indx+1]) not in each_path_edges[id_counter]:
#                         each_path_edges[id_counter].append((path[node_indx],path[node_indx+1]))
#                         each_path_edges[id_counter].append((path[node_indx+1],path[node_indx]))
#                 except:
#                     each_path_edges[id_counter]=[(path[node_indx],path[node_indx+1])]
#                     each_path_edges[id_counter].append((path[node_indx+1],path[node_indx]))
#             id_counter+=1
            
    for commitment_window in [12]:
        #for look_ahead_window in range(4,int(config.look_ahead_window_range)):
        for look_ahead_window in [8]:
            """we first find the candidate paths and use it for action dimention"""
            
            
            game = CFRRL_Game(config, env,commitment_window,look_ahead_window,path_counter,avg_paths_per_time)
            game.set_paths_info(env,config.topology_file,config.each_topology_each_t_each_f_paths)
            """we modify and set the max move to the avg used path at each t for  all flows"""
            game.max_moves = avg_paths_per_time
            #print('this would be the action dimention',path_counter)
            import pdb
            #pdb.set_trace()
            model_weights_queues = []
            experience_queues = []
            if FLAGS.num_agents == 0 or FLAGS.num_agents >= mp.cpu_count():
                FLAGS.num_agents = mp.cpu_count() - 1
            #print('Agent num: %d, iter num: %d\n'%(FLAGS.num_agents+1, FLAGS.num_iter))
            for _ in range(FLAGS.num_agents):
#                 print("we are ddding one to weight queue ")
                model_weights_queues.append(mp.Queue(1))
                experience_queues.append(mp.Queue(1))
                import pdb
            #pdb.set_trace()
            tm_subsets = np.array_split(game.tm_indexes, FLAGS.num_agents)

            coordinator = mp.Process(target=central_agent, args=(config, game,commitment_window,look_ahead_window, model_weights_queues, experience_queues))

            coordinator.start()

            agents = []
            print("these are the agents ",FLAGS.num_agents)
            import pdb
            #pdb.set_trace()
            for i in range(FLAGS.num_agents):
                agents.append(mp.Process(target=agent, args=(i, config, game,commitment_window, look_ahead_window,tm_subsets[i], model_weights_queues[i], experience_queues[i])))

            for i in range(FLAGS.num_agents):
                agents[i].start()

            coordinator.join()


# In[ ]:


if __name__ == '__main__':
    app.run(main)

