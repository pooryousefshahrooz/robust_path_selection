class NetworkConfig(object):
  scale = 1

  max_step = 500 * scale
  
  initial_learning_rate = 0.0001
  learning_rate_decay_rate = 0.96
  learning_rate_decay_step = 5 * scale
  moving_average_decay = 0.9999
  entropy_weight = 0.1

  save_step = 10 * scale
  save_step = 10 
  max_to_keep = 1000

  Conv2D_out = 128
  Dense_out = 128
  
  optimizer = 'RMSprop'
  #optimizer = 'Adam'
    
  logit_clipping = 10       #10 or 0, = 0 means logit clipping is disabled

class Config(NetworkConfig):
  version = 'TE_v2'

  project_name = 'CFR-RL'

  method = 'actor_critic'
  #method = 'pure_policy'
  
  model_type = 'Conv'

  topology_file = 'Abilene'
  traffic_file = 'TM'
  test_traffic_file = 'TM2'
  traffic_file = 'TM3'
  test_traffic_file = 'TM4'
  topology_file = 'ATT_topology_file_modified'
  traffic_file = 'TM3'
  test_traffic_file = 'TM4'
#   topology_file = 'test_topology'
#   traffic_file = 'TM'
#   test_traffic_file = 'TM2'
  tm_history = 1

  max_moves = 20            #percentage

  # For pure policy
  baseline = 'avg'          #avg, best
  commitment_window_range = 8
  look_ahead_window_range = 10
  logging_training_epochs = True
  training_epochs_experiment = True
  printing_flag = True
  each_topology_each_t_each_f_paths= 'each_topology_each_t_each_f_paths.txt'
  testing_results = 'testing_results_final.csv'
  raeke_paths = "raeke_att.txt"
#   raeke_paths = "raeke_abilene.txt"
  capacity_division = 8# 8 for att and 1 for abilene
  demand_scale = 60# 60 for att and 4 for abilene
def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
