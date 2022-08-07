#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import os
import csv
import numpy as np
from absl import app
from absl import flags

import tensorflow as tf
from env import Environment
from game import CFRRL_Game
from model import Network
from config import get_config


# In[ ]:




FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')
toplogy_t_solution_mlu_result = 'toplogy_t_solution_mlu_result.csv'
# topology_name = 'abilene'
time = 0
def sim(config,game):
    time = 0
#     if not os.path.exists(toplogy_t_solution_mlu_result):
#         os.mknod(toplogy_t_solution_mlu_result)
#     with open(toplogy_t_solution_mlu_result, 'a') as newFile:                                
#             newFileWriter = csv.writer(newFile)
#             #print('flow index %s from node %s to the next node %s ratio %s'%(item[0],item[1],item[2],v))
#             newFileWriter.writerow(['topology_name','time/trafic_matrix_id','flow_id','from_node','to_node','ratio','optimal_mlu'])

    for tm_idx in game.tm_indexes:
        _, solution = game.optimal_routing_mlu(tm_idx)
        optimal_mlu, optimal_mlu_delay = game.eval_optimal_routing_mlu(tm_idx, solution, eval_delay=False)
#         with open(toplogy_t_solution_mlu_result, 'a') as newFile:                                
#             newFileWriter = csv.writer(newFile)
#             for item,v in solution.items():
#                 #print('flow index %s from node %s to the next node %s ratio %s'%(item[0],item[1],item[2],v))
#                 newFileWriter.writerow([config.topology_file,time,item[0],item[1],item[2],v,round(optimal_mlu,3)]) #50_04:00:05.606476,0,1000,"3320,6724",link_down_reverse
        time+=1
        
        print('done %s from %s mlu %s'%(time,len(game.tm_indexes),round(optimal_mlu,3)))
        import pdb
        pdb.set_trace()
        #game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay) 

def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env)
    network = Network(config, game.state_dims, game.action_dim, game.max_moves)

    sim(config,game)


if __name__ == '__main__':
    app.run(main)


# In[ ]:


F1= 0.8
F2= 0.838150289017341
# F = 0.8
# Fp = (F*F +1/9 * (1-F) *(1-F))/ (F *F + 2/3 * F*(1-F)+5/9*(1-F) *(1-F))
# print(Fp)


Fp2 = (F1 *F2 + 1/9 *(1-F1)*(1-F2)) / (F1 *F2 + 1/3 *F1*(1-F2)+1/3 *F2*(1-F1)+5/9*(1-F1)*(1-F2))
print(Fp2)


# In[ ]:


F_t1 = 0.8559778176480184
F_store = 0.838150289017341


# In[ ]:


avg_F_approach2 = ((2* 0.8 + 0.8559778176480184+0.8559778176480184)/4+0.8559778176480184)/2
print(avg_F_approach2)


# In[ ]:


(0.8559778176480184  + 0.8) /2


# In[ ]:


(0.838150289017341 + (1 * 0.838150289017341 + 3 * 0.8 )/4 )/2


# In[ ]:


# F_t1 = 0.8559778176480184
# F_t2 = 0.8
# F_approach1 = (0.8+0.8559778176480184)/2
# print(F_approach1)#0.8279889088240092
# print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




