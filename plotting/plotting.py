#!/usr/bin/env python
# coding: utf-8

# In[1]:


from plotting_functions import *
import math
import csv
import time
from math import log


# In[2]:


def get_each_time_demand_ratio():
    each_topology_time_demand_value = {}
    each_topology_traffic_file={'Abilene':'../network_updating_RL/data/AbileneTM','ATT':'../network_updating_RL/data/ATT_topology_file_modifiedTM'}
    topologies = ['ATT','Abilene']
    each_topology_num_nodes = {'Abilene':12,'ATT':25}
    each_topology_dm_in_one_hour={'Abilene':12,'ATT':4}
    each_topology_measurement_granularity={'Abilene':5,'ATT':15}
    
    each_topology_demand_scale={'Abilene':10,'ATT':60}
    
    each_topology_time_demand_value = {}
    avg_each_hour_one_day_demands = {}
    avg_each_interval_one_day_demands= {}
    for topology in topologies:
        granularity = each_topology_measurement_granularity[topology]
        this_time = 0
        each_topology_measured_times = 0
        each_time_demands = {}
        num_nodes = each_topology_num_nodes[topology]
        each_topology_time_demand_value[topology] = {}
        one_hour_interval_counter = 0
        each_topology_one_hour = each_topology_dm_in_one_hour[topology]
        all_sub_hour_interval_demands = []
        f = open(each_topology_traffic_file[topology], 'r')
        traffic_matrices = []
        each_hour_demands = {}
        one_hour_passed = 1
        each_interval_demand = {}
        each_hour_cross_all_days = {}
        avg_each_hour_one_day_demands[topology] = {}
        avg_each_interval_one_day_demands[topology]= {}
        for line in f:
            each_topology_measured_times+=1
            this_time_demands = 0
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            #print("minute ",this_time*granularity)
            #time.sleep(2)
            for v in range(total_volume_cnt):
                i = int(v/num_nodes)
                j = v%num_nodes
                if i != j:
                    #print('from node %s to node %s we have %s trafffic'%(i,j,float(volumes[v])*10))
                    this_time_demands +=float(volumes[v])
            all_sub_hour_interval_demands.append(this_time_demands)
            if one_hour_interval_counter==each_topology_one_hour:
                #print("one hour has passed and we will get the avg of each hour and add it to each hour demands")
                
                one_hour_avg = sum(all_sub_hour_interval_demands)/len(all_sub_hour_interval_demands)/11210296671.674107
                #print("hour %s had %s avg demands"%(one_hour_passed,one_hour_avg/11210296671.674107))
                try:
                    each_hour_demands[one_hour_passed].append(one_hour_avg)
                except:
                    each_hour_demands[one_hour_passed]=[one_hour_avg]
                one_hour_passed +=1
                
                #print("hour is ",one_hour_passed)
                #time.sleep(2)
                
                one_hour_interval_counter = 1
                indx = 0
                #print("and we also add demands for each 5/15 minutes timeinterval")
                #time.sleep(3)
                for each_5_15_interval_demnds in all_sub_hour_interval_demands:
                    try:
                        each_interval_demand[indx].append(each_5_15_interval_demnds)
                    except:
                        each_interval_demand[indx]= [each_5_15_interval_demnds]
                    indx+=1
                all_sub_hour_interval_demands = []
            else:
                one_hour_interval_counter+=1
            if one_hour_passed== 49:
                """ one day passed reset all variables"""
                #print("one day passed and we will get the demands for each hour demand")
            
                one_hour_passed = 1
#                 for each_hour, demands in each_hour_demands.items():
#                     #print("for hour %s we added the demand of that hour to the dictionary for all days"%(each_hour))
#                     try:
#                         each_hour_cross_all_days[each_hour].append(demands)
#                     except:
#                         each_hour_cross_all_days[each_hour]=[demands]
                #each_hour_demands = {}
                all_sub_hour_interval_demands=[]
            this_time+=1
            
            try:
                each_time_demands[this_time].append(this_time_demands)
            except:
                each_time_demands[this_time]=[this_time_demands]

        f.close()
#         if all_sub_hour_interval_demands:
#             one_hour_avg = sum(all_sub_hour_interval_demands)/len(all_sub_hour_interval_demands)
#             try:
#                 each_hour_demands[23].append(one_hour_avg)
#             except:
#                 each_hour_demands[23]=[one_hour_avg]
        alltime_demands = []
        for each_t,demands in each_time_demands.items():
            alltime_demands.append(demands)
        
        
        avg_demands = sum(alltime_demands)/len(alltime_demands)
        #print("***** ",avg_demands)
#         for each_interval,demands in each_interval_demand.items():
#             all_interval_avg_demands = []
#             for demand in demands:
#                 all_interval_avg_demands.append(demand/avg_demands)
#             avg_demands = sum(all_interval_avg_demands)/len(all_interval_avg_demands)
#             avg_each_interval_one_day_demands[topology][each_interval]=avg_demands
        for each_hour,this_demand_ratios in each_hour_demands.items():
#             all_hour_avg_demands = []
#             for demand in this_demands:
#                 all_hour_avg_demands.append(demand/avg_demands)
            avg_demands = sum(this_demand_ratios)/len(this_demand_ratios)
            #print("for hour %s we have %s rate for demands"%(each_hour,avg_demands))
            avg_each_hour_one_day_demands[topology][each_hour]=avg_demands
        
#         print('for topology %s we have %s traffic matrices '%(topology,each_topology_measured_times))
#         
            
        
#         for each_t,demands in each_time_demands.items():
#             ratio = demands/avg_demands
#             each_topology_time_demand_value[topology][each_t] = ratio
    avg_each_hour_one_day_demands['Abilene'][0]=avg_each_hour_one_day_demands['Abilene'][1]
    avg_each_hour_one_day_demands['ATT'][0]=avg_each_hour_one_day_demands['ATT'][1]
    return avg_each_interval_one_day_demands,list(avg_each_interval_one_day_demands[topology].keys()),avg_each_hour_one_day_demands,list(avg_each_hour_one_day_demands.keys())


# In[ ]:


def add_item(each_scheme_each_epoch_each_commitment_lookahead_mlu_value,epoch,commitment,scheme,lookahead,ratio):
    try:
        each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme][lookahead].append(ratio)
    except:
        try:
            each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme][lookahead]=[ratio]
        except:
            try:
                each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme]={}
                each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme][lookahead]=[ratio]
            except:
                try:
                    each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment]={}
                    each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme]={}
                    each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme][lookahead]=[ratio]
                except:
                    try:
                        each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch]={}
                        each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment]={}
                        each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme]={}
                        each_scheme_each_epoch_each_commitment_lookahead_mlu_value[epoch][commitment][scheme][lookahead]=[ratio]
                    except ValueError:
                        print(ValueError)
    return each_scheme_each_epoch_each_commitment_lookahead_mlu_value


# In[ ]:


file_result_path = '../network_updating_RL/testing_results_test.csv'
# [topology_file,commitment_window,look_ahead_window,time_index,
#  "optimal",item[0],item[1],item[2],v,round(hindsight_mlus[mlu_index],3),
# "mlu_optimal",item[0],item[1],item[2],mlu_optimal_solutions[solution_indx][(item[0],item[1],item[2])],round(hindsight_mlus[mlu_index],3),
# "rl",item[0],item[1],item[2],rl_solutions[solution_indx][(item[0],item[1],item[2])],round(rl_mlus[mlu_index],3),
# "ecmp",round(ecmp_mlus[mlu_index],3),"oblivious",round(oblivious_mlus[mlu_index],3)]
schemes = ["ECMP","Oblivious","MLU optimal","DRL"]
line_counter = 0
for topology in ["Abilene"]:
    epoch_numbers = []
    each_scheme_each_commitment_lookahead_mlu_value = {}
    commitments = []
    look_aheads = []
    with open(file_result_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in (reader):
            print("we done %s from %s"%(line_counter,100374120))
            line_counter+=1
            if line[0] in topology:
                scheme = line[0]
                commitment = int(line[1])
                lookahead = int(line[2])
                
                if commitment not in commitments:
                    commitments.append(commitment)
                if lookahead not in look_aheads:
                    look_aheads.append(lookahead)
                
                optimal_mlu = float(line[9])
                mlu_optimal_ratio = float(line[15])/optimal_mlu
                drl_ratio = float(line[21])/optimal_mlu
                ecmp_ratio = float(line[23])/optimal_mlu
                oblivious_ratio = float(line[25])/optimal_mlu
                epoch_number = int(line[26])
                if epoch_number not in epoch_numbers:
                    epoch_numbers.append(epoch_number)
                each_scheme_each_epoch_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,epoch_number,commitment,'ECMP',lookahead,ecmp_ratio)
                
                each_scheme_each_epoch_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,epoch_number,commitment,'Oblivious',lookahead,oblivious_ratio)        
                each_scheme_each_epoch_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,epoch_number,commitment,'DRL',lookahead,drl_ratio)
                each_scheme_each_epoch_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,epoch_number,commitment,"MLU optimal",lookahead,mlu_optimal_ratio)
    #print("each_scheme_each_epoch_commitment_lookahead_mlu_value",each_scheme_each_epoch_commitment_lookahead_mlu_value)
    print("commitments ",commitments)
    print("look_aheads",look_aheads)
    print("schemes",schemes)
    print("epoch_numbers",epoch_numbers)
    for commitment in commitments:
        for look_ahead in look_aheads:
            each_scheme_each_epoch_mlu_value = {}
            for scheme in schemes:
                for epoch in epoch_numbers:
                    ratios = each_scheme_each_commitment_lookahead_mlu_value[epoch][commitment][scheme][look_ahead]
                    avg_ratio = sum(ratios)/len(ratios)
                    try:
                        each_scheme_each_epoch_mlu_value[scheme][epoch] = avg_ratio
                    except:
                        each_scheme_each_epoch_mlu_value[scheme]={}
                        each_scheme_each_epoch_mlu_value[scheme][epoch] = avg_ratio
                    with open("plotting_scheme_epoch_mlu_ratio.csv", 'a') as newFile:                                
                        newFileWriter = csv.writer(newFile)
                        newFileWriter.writerow([commitment,look_ahead,scheme,epoch,avg_ratio])
            print('each_scheme_each_commitment_lookahead_mlu_value',each_scheme_each_commitment_lookahead_mlu_value)

            ploting_simple_y_as_x('Number of epochs','MLU ratio',schemes,
                                each_scheme_each_epoch_mlu_value,
                                  epoch_numbers,epoch_numbers,False,
                                  'plots/mlu_as_a_function_of_epoch_number_'+topology+str(commitment)+str(look_ahead)+'.pdf')
    
#     for commitment in commitments: 
#         for window in look_aheads:
            
#         ploting_simple_y_as_x('Look ahead window','MLU ratio',schemes,
#                         each_scheme_each_commitment_lookahead_mlu_value[commitment],
#                           look_aheads,look_aheads,False,'plots/mlu_as_a_function_of_look_ahead_'+topology+str(commitment)+'.pdf')
        
import pdb
pdb.set_trace()


# In[ ]:





# In[ ]:


# each_topology_time_demand_value= {'Abilene':{0:1.1,1:1.2,2:1.5,3:1.5,4:1.3,5:1.1,6:1.1},
#                                'ATT':{0:0.7,1:0.8,2:1.0,3:0.8,4:0.7,5:0.6,6:0.6}}
# topologies_in_order = ['ATT','Abilene']
# value_for_x_axis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
#                    25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
# avg_each_interval_one_day_demands,all_intervals,avg_each_hour_one_day_demands,all_hours =get_each_time_demand_ratio()
# print('each_topology_time_demand_value',avg_each_hour_one_day_demands)
# ploting_simple_y_as_x('Time (hour)','Demand ratio',topologies_in_order,avg_each_hour_one_day_demands,value_for_x_axis,value_for_x_axis,False,'plots/load_time_result_both.pdf')


# In[ ]:


# each_topology_time_demand_value= {'Abilene':{0:1.1,1:1.2,2:1.5,3:1.5,4:1.3,5:1.1,6:1.1},
#                                'ATT':{0:0.7,1:0.8,2:1.0,3:0.8,4:0.7,5:0.6,6:0.6}}
# each_topology_time_demand_value = {'Abilene': 
#                                    {1: 1.043780407884243, 2: 1.0683178142556684, 
#                                     3: 1.0230616163652722, 4: 1.001849883994401,
#                                     5: 0.9922586620117816, 6: 0.957841000912626,
#                                     7: 0.9328748439600039, 8: 0.9260538389817884,
#                                     9: 0.9108001908809503, 10: 0.8835988281036701, 
#                                     11: 0.8672667086506801, 12: 0.8210314838686502,
#                                     13: 0.8003642148030181, 14: 0.8343584568479496, 
#                                     15: 0.9009250034938633, 16: 0.9745626573941767, 
#                                     17: 1.0657542560433462, 18: 1.1008907775516772, 
#                                     19: 1.142110926872586, 20: 1.1751101819166412, 
#                                     21: 1.1798108080565881, 22: 1.157839485767383, 
#                                     23: 1.1477348014909956, 24: 1.1332807259021045}}

# topologies_in_order = ['Abilene']
# value_for_x_axis = [0,1,2,3,4,5,6]
# print('each_topology_time_demand_value',each_topology_time_demand_value)
# ploting_simple_y_as_x('time','Demand ratio',topologies_in_order,each_topology_time_demand_value,value_for_x_axis,value_for_x_axis,False,'plots/load_time_result_abilene.pdf')


# In[ ]:



file_result_path = '../network_updating_RL/testing_results.csv'
# [topology_file,commitment_window,look_ahead_window,time_index,
#  "optimal",item[0],item[1],item[2],v,round(hindsight_mlus[mlu_index],3),
# "mlu_optimal",item[0],item[1],item[2],mlu_optimal_solutions[solution_indx][(item[0],item[1],item[2])],round(hindsight_mlus[mlu_index],3),
# "rl",item[0],item[1],item[2],rl_solutions[solution_indx][(item[0],item[1],item[2])],round(rl_mlus[mlu_index],3),
# "ecmp",round(ecmp_mlus[mlu_index],3),"oblivious",round(oblivious_mlus[mlu_index],3)]
schemes = ["ECMP","Oblivious","MLU optimal","DRL"]
line_counter = 0
for topology in ["Abilene"]:
    each_scheme_each_commitment_lookahead_mlu_value = {}
    commitments = []
    look_aheads = []
    with open(file_result_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in (reader):
            #print("we done %s from %s"%(line_counter,93091680))
            line_counter+=1
            if line[0] in topology:
                scheme = line[0]
                commitment = int(line[1])
                lookahead = int(line[2])
                
                if commitment not in commitments:
                    commitments.append(commitment)
                if lookahead not in look_aheads:
                    look_aheads.append(lookahead)
                
                optimal_mlu = float(line[9])
                mlu_optimal_ratio = float(line[15])/optimal_mlu
                drl_ratio = float(line[21])/optimal_mlu
                ecmp_ratio = float(line[23])/optimal_mlu
                oblivious_ratio = float(line[25])/optimal_mlu
                
                each_scheme_each_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,commitment,'ECMP',lookahead,ecmp_ratio)
                
                each_scheme_each_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,commitment,'Oblivious',lookahead,oblivious_ratio)        
                each_scheme_each_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,commitment,'DRL',lookahead,drl_ratio)
                each_scheme_each_commitment_lookahead_mlu_value = add_item(
                    each_scheme_each_commitment_lookahead_mlu_value,commitment,"MLU optimal",lookahead,mlu_optimal_ratio)
                
    for commitment in commitments:
        for look_ahead in look_aheads:
            for scheme in schemes:
                ratios = each_scheme_each_commitment_lookahead_mlu_value[commitment][scheme][look_ahead]
                avg_ratio = sum(ratios)/len(ratios)
                each_scheme_each_commitment_lookahead_mlu_value[commitment][scheme][look_ahead] = avg_ratio
    print('each_scheme_each_commitment_lookahead_mlu_value',each_scheme_each_commitment_lookahead_mlu_value)
    for commitment in commitments:        
        ploting_simple_y_as_x('Look ahead window','MLU ratio',schemes,
                        each_scheme_each_commitment_lookahead_mlu_value[commitment],
                          look_aheads,look_aheads,False,'plots/mlu_as_a_function_of_look_ahead_'+topology+str(commitment)+'.pdf')


# In[ ]:



for commitment in commitments:        
    ploting_simple_y_as_x('Look ahead window,c='+str(commitment),'MLU ratio',schemes,
                    each_scheme_each_commitment_lookahead_mlu_value[commitment],
                      look_aheads,look_aheads,False,'plots/mlu_as_a_function_of_look_ahead_'+topology+'.pdf')


# In[ ]:


import pdb
pdb.set_trace()


# In[ ]:


my_list = [1,2,34]
my_list.pop(0)
print(my_list)
for node_indx in range(1-1):
    print("first")


# In[ ]:


file_result_path = ''
# [topology_file,commitment_window,look_ahead_window,time_index,
#  "optimal",item[0],item[1],item[2],v,round(hindsight_mlus[mlu_index],3),
# "mlu_optimal",item[0],item[1],item[2],mlu_optimal_solutions[solution_indx][(item[0],item[1],item[2])],round(hindsight_mlus[mlu_index],3),
# "rl",item[0],item[1],item[2],rl_solutions[solution_indx][(item[0],item[1],item[2])],round(rl_mlus[mlu_index],3),
# "ecmp",round(ecmp_mlus[mlu_index],3),"oblivious",round(oblivious_mlus[mlu_index],3)]
for tpology in ["Abilene", "ATT"]:
    each_topology_each_scheme_each_commitment_lookahead_mlu_value
    with open(file_result_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in (reader):
            if line[0] in topology:
                scheme = line[0]
                commitment = int(line[1])
                lookahead = int(line[2])
                optimal_mlu = float(line[9])
                mlu_optimal_ratio = float(line[15])/optimal_mlu
                drl_ratio = float(line[21])/optimal_mlu
                ecmp_ratio = float(line[23])/optimal_mlu
                oblivious_ratio = float(line[25])/optimal_mlu
                
                
                
                each_scheme_each_commitment_lookahead_mlu_value[lookahead]['ECMP'][commitment].append(ecmp_ratio)
                each_scheme_each_commitment_lookahead_mlu_value[lookahead]['Oblivious'][commitment].append(oblivious_ratio)
                each_scheme_each_commitment_lookahead_mlu_value[lookahead]['DRL'][commitment].append(drl_ratio)
                each_scheme_each_commitment_lookahead_mlu_value[lookahead]["MLU optimal"][commitment].append(mlu_optimal_ratio)
        
    for look_ahead in look_aheads:
        for commitment in commitments:
            for scheme in schemes:
                ratios = each_scheme_each_commitment_lookahead_mlu_value[look_ahead][scheme][commitment]
                avg_ratio = sum(ratios)/len(ratios)
                each_scheme_each_commitment_lookahead_mlu_value[look_ahead][scheme][commitment] = avg_ratio
    print('each_scheme_each_commitment_lookahead_mlu_value',each_scheme_each_commitment_lookahead_mlu_value)
    for look_ahead in look_aheads:        
        ploting_simple_y_as_x('Look-ahead window','MLU ratio',schemes,each_scheme_each_commitment_lookahead_mlu_value[look_ahead],
                          commitment_values,commitment_values,False,'plots/mlu_as_a_function_of_commitment_'+tpology+'.pdf')


# In[ ]:


each_approach_potential_events_fraction = {'random':[10,15,12,15,10,11,5,6,5,6,5,6,5,6,5,6,5,6,5,10,20,30,24,15,11,13,14,15,16,20,40,10,10,10]}
cdf_info_dictionary_over_multi_item = {}
cdf_info_dictionary_over_multi_item['random'] = {}
for scheme, reductions in each_approach_potential_events_fraction.items():
    #cdf_info_dictionary_over_multi_item[scheme][0] = 0
    for RD in reductions:
        try:
            cdf_info_dictionary_over_multi_item[scheme][RD] =cdf_info_dictionary_over_multi_item[scheme][RD] +1
        except:
            cdf_info_dictionary_over_multi_item[scheme][RD] = 1

list_of_keys = list(each_approach_potential_events_fraction.keys())
min_value = 0
max_value = 0
for scheme, reductions in each_approach_potential_events_fraction.items():
    print(scheme,min(reductions), max(reductions),reductions)
    if min(reductions)<min_value:
        min_value = min(reductions)
    if  max(reductions)>max_value:
        max_value =  max(reductions)
y_min_value= 0.0
y_max_value = 1.0

multiple_lines_cdf("Fraction of events that have necessary condition",'cumulative fraction \n of sub topologies',cdf_info_dictionary_over_multi_item,False,'plots/Distribution_on_subtopology_events.pdf',list_of_keys,min_value,max_value,y_min_value,y_max_value)


# In[ ]:



potentiality_of_getting_benefit={
                                    'p=4':[10,14,20,20,22,23,24,25],
                                    'p=10':[10,16,20,25,34,35,36,37],
                                    'p=20':[10,16,20,25,34,35,36,37]}
plot_bar_plot('Propagation delay(ms)','Sent messages',potentiality_of_getting_benefit,'plots/CD_reduction.pdf')
rho = '\circ'
test = r'$\alpha$'

print (rho,test)
# cross_all_topologies_table_sizes = ['[0-1]','4','15k','20k','25k','30k','80k','130k']
# print ('cross_all_topologies_table_sizes',cross_all_topologies_table_sizes)
# plot_bar_chart('Propagation delay','Fraction of events',table_sizes_dictionary,['Not needed','Potential'],cross_all_topologies_table_sizes,'plots/Potentiality_of_MRAI.pdf',False)


# In[ ]:





# In[ ]:



each_approach_each_x_axis_pont_values = {'Work-conserving':{
                                        0:[0.6,0.7,0.7,0.6,0.7,1.4,1.2],
                                        10:[0.6,0.7,0.7,0.6,0.7,1.4,1.2], 
                                        20: [0.7,0.7,0.6,0.7,0.6,0.4,1.1], 
                                        30:[0.8,0.9,0.7,1.2,0.7,1.1]},
                                         'Optimal MRAI':{
                                            0:[0.5,0.5,0.4,0.6,0.5,0.3,0.1,0.1], 
                                             10:[0.5,0.5,0.4,0.6,0.5,0.3,0.1,0.1], 
                                             20: [0.6,0.6,0.4,0.7,0.8,0.9,1.0], 
                                             30:[1.0,0.9,1.0,0.8,0.9,0.7]},
                                         
}


# each_approach_each_x_axis_pont_values = {'Work-conserving':{
#                                         0:[0.5,0.6,0.7,0.5,0.6,0.2,0.0], 
#                                         10: [0.5,0.3,0.4,0.4,0.6,0.2], 
#                                         20:[0.3,0.2,0.3,0.2,0.1,0.0]},
#                                          'Optimal MRAI':{
#                                         0:[0.4,0.3,0.4,0.4,0.6,0.3,0.1,0.0], 
#                                              10: [0.3,0.3,0.2,0.3,0.5,0.2,0.0], 
#                                              20:[0.1,0.1,0.3,0.1,0.0,1.0]},
                                         
# }


# each_approach_each_x_axis_pont_values = {'Work-conserving':{
#                                         0:[0.4,0.5,0.6,0.5,0.6,0.1,0.0], 
#                                         10: [0.4,0.3,0.3,0.4,0.6,0.0], 
#                                         20:[0.2,0.2,0.2,0.1,0.0,0.0]},
#                                          'Optimal MRAI':{
#                                         0:[0.3,0.3,0.4,0.3,0.6,0.2,0.0,0.0], 
#                                              10: [0.2,0.3,0.2,0.2,0.5,0.0,0.0], 
#                                              20:[0.1,0.1,0.2,0.1,0.0,0.0]},
                                         
# }

tickets_on_x_axis = ['[1-1]','[1-10]','[10-50]','[50-100]']

# positions [-0.4  1.6  3.6]
multiple_box_plot_on_each_x_axis('Propagation delay(ms)','Normalized CD',tickets_on_x_axis,[0,10,20,30],each_approach_each_x_axis_pont_values,'plots/normalized_CD_propagation_dealy_p=4.pdf')


# In[ ]:


each_approach_each_x_axis_pont_values = {'Work-conserving':{
                                        0:[2,1,1,2,1,2,0.0,0,2,0,1,1,2,1,2,2,2,2,0,1,2,3,2], 
                                        10: [1,0,1,1,1,1,0.0,0,0,0,1,0,0,1,1,2,1,0,0,1,1,2,2], 
                                        20:[2,1,1,0,1,1,0.0,0,0,0,1,1,1,1,1,1,2,2,0,1,1,1,2]},
                                                                                  
}

# each_approach_each_x_axis_pont_values = {'Work-conserving':{
#                                         0:[0.5,0.6,0.7,0.5,0.6,0.2,0.0], 10: [0.5,0.3,0.4,0.4,0.6,0.2], 20:[0.3,0.2,0.3,0.2,0.1,0.0]},
#                                          'Optimal MRAI':{
#                                         0:[0.4,0.3,0.4,0.4,0.6,0.3,0.1,0.0], 10: [0.3,0.3,0.2,0.3,0.5,0.2,0.0], 20:[0.1,0.1,0.3,0.1,0.0,1.0]},
                                         
# }


# each_approach_each_x_axis_pont_values = {'Work-conserving':{
#                                         0:[0.6,0.7,0.7,0.6,0.7,0.4,0.2], 10: [0.6,0.5,0.6,0.7,0.6,0.4,0.4], 20:[0.4,0.4,0.3,0.2,0.3,0.2]},
#                                          'Optimal MRAI':{
#                                         0:[0.5,0.5,0.4,0.6,0.5,0.3,0.1,0.1], 10: [0.4,0.5,0.4,0.3,0.6,0.2,0.1], 20:[0.2,0.3,0.4,0.3,0.2,0.0]},
                                         
# }
tickets_on_x_axis = ['[1-10]','[10-50]','[50-100]']

# positions [-0.4  1.6  3.6]
multiple_box_plot_on_each_x_axis('Propagation delay(ms)','Average queue size',tickets_on_x_axis,[0,10,20],each_approach_each_x_axis_pont_values,'plots/queuing_size_propagation_dealy_p=5.pdf')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def check_event_in_both_propagations(event,each_approach_each_propagation_processing_delay_MRAI_CD,propagations):
    
    for processing, propagation_CD in each_approach_each_propagation_processing_delay_MRAI_CD[event].items():
        for propagation in propagations:
            if propagation not in propagation_CD.keys():
                return False
    return True
file_result_path = 'each_event_info_result.csv'
each_event_ID = {}
each_approach_each_propagation_processing_delay_MRAI_CD = {} 
MRAI_values = []
event_counter = 0
propagation_values = []

        
with open(file_result_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in (reader):
        propagation_delay = line[8]
        procesing_delay= line[7]
        MRAI_value = int(line[3])
        if MRAI_value not in MRAI_values:
            MRAI_values.append((MRAI_value))
        CD = float(line[9])
#         CD = float(line[10])
        approach = line[1]
        event_title = line[0]
#         if 'Work' in approach or 'work' in approach:
#             approach = 'Work-conserving'
#         if 'MRAI' in approach:
#             approach = 'Optimal MRAI'
        if 'up' in line[0]:
    #             approach = approach+', UP phase'
    #         else:
    #             approach = approach+', DOWN phase'
            approach = approach
            try:
                each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value].append(CD)
            except:
                try:
                    each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                except:
                    try:
                        each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                        each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                    except:
                        try:
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                        except:
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                            
for approach, each_propagation_processing_delay_MRAI_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
    for propagation_delay,processing_delay_MRAI_CDs in each_propagation_processing_delay_MRAI_CD.items():
        for processing_delay,MRAI_CDs in processing_delay_MRAI_CDs.items():
            for MRAI,CDs in MRAI_CDs.items():
                each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][processing_delay][MRAI] = sum(CDs)/len(CDs)


# processing_delays = ['4','20','50']
# propagation_delays = [1]
# for event_approach , processing_propagation_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
#     for processing, propagation_CD in processing_propagation_CD.items():
#         print(event_approach,processing,propagation_CD)
each_approach_processing_each_propagation_convergence_delay_reduction = {}
# print('MRAI_values',MRAI_values)
propagation_delays = []
new_x_axis_tickets = []
each_processing_scheme_MRAI_CD_value = {}
MRAI_value_for_x_axis = []
processing_delay_as_schemes_in_order = []
for approach, each_propagation_processing_delay_MRAI_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
    for each_propagation_delay,each_processing_delay_MRAI_CD in each_propagation_processing_delay_MRAI_CD.items():
        
        processing_delays = list(each_processing_delay_MRAI_CD.keys())
        if str(each_propagation_delay) not in propagation_delays:
            propagation_delays.append(str(each_propagation_delay))
        if '[1-'+str(each_propagation_delay)+']' not in new_x_axis_tickets:
            new_x_axis_tickets.append('[1-'+str(each_propagation_delay)+']')
        schemes_in_order = processing_delays
    #     each_processing_delay_MRAI_CD = {'4':{0:4,5:2,10:6},
    #                                     '20':{0:4,5:2,10:6},
    #                                     '50':{0:4,5:2,10:6},}
        MRAI_values.sort()
       # print('each_processing_delay_MRAI_CD ',each_processing_delay_MRAI_CD)
        #print('MRAI_values ',MRAI_values)

        tickets_on_x_axis = ['[1-10]','[10-50]','[50-100]']
        tickets_on_x_axis =MRAI_values

        # positions [-0.4  1.6  3.6]

        #multiple_box_plot_on_each_x_axis('Propagation delay(ms)','Average queue size',tickets_on_x_axis,[0,10,20],each_approach_each_x_axis_pont_values,'plots/queuing_size_propagation_dealy_p=5.pdf')
        for processing_delay,MRAI_CD in each_processing_delay_MRAI_CD.items():

            convergence_delays = []
            if approach+',t='+str(processing_delay)+'\u03BCs' not in processing_delay_as_schemes_in_order:
                processing_delay_as_schemes_in_order.append(approach+',t='+str(processing_delay)+'\u03BCs')
            for MRAI, CD in MRAI_CD.items():
                if MRAI not in MRAI_value_for_x_axis:
                    MRAI_value_for_x_axis.append(MRAI)
                convergence_delays.append(CD)
                try:
                    each_processing_scheme_MRAI_CD_value[approach+',t='+str(processing_delay)+'\u03BCs'][MRAI] = CD
                except:
                    each_processing_scheme_MRAI_CD_value[approach+',t='+str(processing_delay)+'\u03BCs']={}
                    each_processing_scheme_MRAI_CD_value[approach+',t='+str(processing_delay)+'\u03BCs'][MRAI] = CD   

        MRAI_value_for_x_axis.sort()
ploting_simple_y_as_x('MRAI(msec) d='+str(each_propagation_delay),'CD(ms) ',processing_delay_as_schemes_in_order,each_processing_scheme_MRAI_CD_value,MRAI_value_for_x_axis,MRAI_value_for_x_axis,False,'plots/CD_MRAI_nodes.pdf')


# In[ ]:


def check_event_in_both_propagations(event,each_approach_each_propagation_processing_delay_MRAI_CD,propagations):
    
    for processing, propagation_CD in each_approach_each_propagation_processing_delay_MRAI_CD[event].items():
        for propagation in propagations:
            if propagation not in propagation_CD.keys():
                return False
    return True
file_result_path = 'each_event_info_result.csv'
each_event_ID = {}
each_approach_each_propagation_processing_delay_MRAI_CD = {} 
MRAI_values = []
event_counter = 0
propagation_values = []

        
with open(file_result_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in (reader):
        propagation_delay = line[8]
        procesing_delay= line[7]
        MRAI_value = int(line[3])
        if MRAI_value not in MRAI_values:
            MRAI_values.append((MRAI_value))
        CD = float(line[9])
#         CD = float(line[10])
        approach = line[1]
        event_title = line[0]
        if 'Work' in approach or 'work' in approach:
            approach = 'Work-conserving'
        if 'MRAI' in approach:
            approach = 'Optimal MRAI'
        if 'up' in line[0]:
    #             approach = approach+', UP phase'
    #         else:
    #             approach = approach+', DOWN phase'
            approach = approach
            try:
                each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value].append(CD)
            except:
                try:
                    each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                except:
                    try:
                        each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                        each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                    except:
                        try:
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                        except:
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                            
for approach, each_propagation_processing_delay_MRAI_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
    for propagation_delay,processing_delay_MRAI_CDs in each_propagation_processing_delay_MRAI_CD.items():
        for processing_delay,MRAI_CDs in processing_delay_MRAI_CDs.items():
            for MRAI,CDs in MRAI_CDs.items():
                each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][processing_delay][MRAI] = sum(CDs)/len(CDs)


# processing_delays = ['4','20','50']
# propagation_delays = [1]
# for event_approach , processing_propagation_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
#     for processing, propagation_CD in processing_propagation_CD.items():
#         print(event_approach,processing,propagation_CD)
each_approach_processing_each_propagation_convergence_delay_reduction = {}
# print('MRAI_values',MRAI_values)
propagation_delays = []
new_x_axis_tickets = []
for approach, each_propagation_processing_delay_MRAI_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
    for each_propagation_delay,each_processing_delay_MRAI_CD in each_propagation_processing_delay_MRAI_CD.items():
        processing_delay_as_schemes_in_order = []
        processing_delays = list(each_processing_delay_MRAI_CD.keys())
        if str(each_propagation_delay) not in propagation_delays:
            propagation_delays.append(str(each_propagation_delay))
        if '[1-'+str(each_propagation_delay)+']' not in new_x_axis_tickets:
            new_x_axis_tickets.append('[1-'+str(each_propagation_delay)+']')
        schemes_in_order = processing_delays
    #     each_processing_delay_MRAI_CD = {'4':{0:4,5:2,10:6},
    #                                     '20':{0:4,5:2,10:6},
    #                                     '50':{0:4,5:2,10:6},}
        MRAI_values.sort()
       # print('each_processing_delay_MRAI_CD ',each_processing_delay_MRAI_CD)
        #print('MRAI_values ',MRAI_values)

        #tickets_on_x_axis = ['[1-10]','[10-50]','[50-100]']
        tickets_on_x_axis =MRAI_values
        each_processing_scheme_MRAI_CD_value = {}
        MRAI_value_for_x_axis = []
        # positions [-0.4  1.6  3.6]

        #multiple_box_plot_on_each_x_axis('Propagation delay(ms)','Average queue size',tickets_on_x_axis,[0,10,20],each_approach_each_x_axis_pont_values,'plots/queuing_size_propagation_dealy_p=5.pdf')
        for processing_delay,MRAI_CD in each_processing_delay_MRAI_CD.items():

            zero_batch_CD = each_approach_each_propagation_processing_delay_MRAI_CD['Optimal MRAI'][each_propagation_delay][processing_delay][1]
            convergence_delays = []
            if 'p='+str(processing_delay) not in processing_delay_as_schemes_in_order:
                processing_delay_as_schemes_in_order.append('p='+str(processing_delay))
            for MRAI, CD in MRAI_CD.items():
                if MRAI not in MRAI_value_for_x_axis:
                    MRAI_value_for_x_axis.append(MRAI)
                if 'Work-conserving' in approach:
                    if int(MRAI)==1000:
                        convergence_delays.append(CD)
                else:
                    if int(MRAI)!=1000:
                        convergence_delays.append(CD)
                if 'Optimal MRAI' in approach:
                    if int(MRAI)!=1000:
                        try:
                            each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD
                        except:
                            each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)]={}
                            each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD
                else:
                    try:
                        each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD
                    except:
                        each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)]={}
                        each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD   

            convergence_delay_reduction = (zero_batch_CD-min(convergence_delays))/zero_batch_CD
            convergence_delay_reduction =min(convergence_delays)/zero_batch_CD
            
            print('scheme %s  d %s min(CDs) %s zero CD %s   result %s'%(approach+', p='+processing_delay+'\u03BCs',each_propagation_delay,min(convergence_delays),zero_batch_CD,convergence_delay_reduction))
            try:
                each_approach_processing_each_propagation_convergence_delay_reduction[approach+', p='+processing_delay+'\u03BCs'][int(each_propagation_delay)] = convergence_delay_reduction
            except:
                each_approach_processing_each_propagation_convergence_delay_reduction[approach+', p='+processing_delay+'\u03BCs'] = {}
                each_approach_processing_each_propagation_convergence_delay_reduction[approach+', p='+processing_delay+'\u03BCs'][int(each_propagation_delay)] = convergence_delay_reduction
        MRAI_value_for_x_axis.sort()
        ploting_simple_y_as_x('MRAI(msec) d='+str(each_propagation_delay),'CD(ms) ',processing_delay_as_schemes_in_order,each_processing_scheme_MRAI_CD_value,MRAI_value_for_x_axis,MRAI_value_for_x_axis,False,'plots/CD_focus_MRAI_processing_delay_propagation_'+str(each_propagation_delay)+'.pdf')

schemes_in_order = []
processing_delays.sort()
for approach in each_approach_each_propagation_processing_delay_MRAI_CD:
    for processing_delay in processing_delays:
        schemes_in_order.append(approach+', p='+str(processing_delay)+'\u03BCs')
# print('each_processing_each_propagation_convergence_delay_reduction ',each_approach_processing_each_propagation_convergence_delay_reduction)
# print('schemes_in_order ',schemes_in_order)
# print('propagation_delays ',propagation_delays)
plot_convergence_detection_alg_overhead('Propagation delay range(msec): log scale','Normalized CD',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,new_x_axis_tickets,False,'plots/normalized_CD_real_internet_subtopologies_propagation_delay.pdf')


# In[ ]:



tickets_on_x_axis = ['[1-1]','[1-10]']
each_approach_each_x_axis_point_values = {'Work-conserving':{
                                        0:[0.6,0.7,0.7,0.6,0.7,1.4,1.2],
                                        10:[0.6,0.7,0.7,0.6,0.7,1.4,1.2], 
                                        20: [0.7,0.7,0.6,0.7,0.6,0.4,1.1], 
                                        30:[0.8,0.9,0.7,1.2,0.7,1.1]},
                                         'Optimal MRAI':{
                                            0:[0.5,0.5,0.4,0.6,0.5,0.3,0.1,0.1], 
                                             10:[0.5,0.5,0.4,0.6,0.5,0.3,0.1,0.1], 
                                             20: [0.6,0.6,0.4,0.7,0.8,0.9,1.0], 
                                             30:[1.0,0.9,1.0,0.8,0.9,0.7]},
                                         
}
# positions [-0.4  1.6  3.6]


def check_event_in_both_propagations(event,each_approach_each_propagation_processing_delay_MRAI_CD,propagations):
    
    for processing, propagation_CD in each_approach_each_propagation_processing_delay_MRAI_CD[event].items():
        for propagation in propagations:
            if propagation not in propagation_CD.keys():
                return False
    return True
file_result_path = 'each_event_info_result.csv'
each_event_ID = {}
each_approach_each_propagation_processing_delay_MRAI_CD = {} 
MRAI_values = []
event_counter = 0
propagation_values = []
each_event_each_approach_each_processing_propagation_CDs = {}
processing_delays = []
propagation_delays = []
with open(file_result_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in (reader):
        propagation_delay = line[8]
        if propagation_delay not in propagation_delays:
            propagation_delays.append(propagation_delay)
        procesing_delay= line[7]
        if procesing_delay not in processing_delays:
            processing_delays.append(procesing_delay)
        MRAI_value = int(line[3])
        if MRAI_value not in MRAI_values:
            MRAI_values.append((MRAI_value))
        CD = float(line[9])
#         CD = float(line[10])
        approach = line[1]
        event_title = line[0]
        if 'up' in event_title or 1==1:
            if 'Work' in approach or 'work' in approach:
                approach = 'Work-conserving'
            if 'MRAI' in approach or 'optimal' in approach:
                approach = 'Optimal MRAI'
    #         if 'up' in line[0]:
    #             approach = approach+', UP phase'
    #         else:
    #             approach = approach+', DOWN phase'
            approach = approach
            try:
                each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay].append(CD)
            except:
                try:
                    each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay]= [CD]
                except:
                    try:
                        each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay] = {}
                        each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay] = [CD]
                    except:
                        try:
                            each_event_each_approach_each_processing_propagation_CDs[event_title][approach] = {}
                            each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay] = {}
                            each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay] = [CD]
                        except:
                            try:
                                each_event_each_approach_each_processing_propagation_CDs[event_title] = {}
                                each_event_each_approach_each_processing_propagation_CDs[event_title][approach] = {}
                                each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay] = {}
                                each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay] = [CD]
                            except ValueError:
                                print('ValueError',ValueError)


            if 'MRAI' in approach:
                
                approach = 'zero_batching'
                try:
                    each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay].append(CD)
                except:
                    try:
                        each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay]= [CD]
                    except:
                        try:
                            each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay] = {}
                            each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay] = [CD]
                        except:
                            try:
                                each_event_each_approach_each_processing_propagation_CDs[event_title][approach] = {}
                                each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay] = {}
                                each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay] = [CD]
                            except:
                                try:
                                    each_event_each_approach_each_processing_propagation_CDs[event_title] = {}
                                    each_event_each_approach_each_processing_propagation_CDs[event_title][approach] = {}
                                    each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay] = {}
                                    each_event_each_approach_each_processing_propagation_CDs[event_title][approach][procesing_delay][propagation_delay] = [CD]
                                except ValueError:
                                    print('ValueError',ValueError)

each_approach_processing_each_propagation_convergence_delay_reduction = {}
print('MRAI_values',MRAI_values)

new_x_axis_tickets = []

        #ploting_simple_y_as_x('MRAI(msec) d='+str(each_propagation_delay),'CD(ms) ',processing_delay_as_schemes_in_order,each_processing_scheme_MRAI_CD_value,MRAI_value_for_x_axis,MRAI_value_for_x_axis,False,'plots/CD_focus_MRAI_processing_delay_propagation_'+str(each_propagation_delay)+'.pdf')
print('done')
apporaches = ['Optimal MRAI','Work-conserving']
print('processing_delays',processing_delays)
print('propagation_delays',propagation_delays)
for processing_delay in processing_delays:
    each_approach_each_x_axis_point_values = {}
    print('for processing delay ',processing_delay)
    for each_approach in apporaches:
        print('for approach ',each_approach)
        for propagation_delay in propagation_delays:
            print('for propagation delay ',propagation_delay)
            list_of_normalized_CDs = []
            for event,each_approach_each_processing_propagation_CDs in each_event_each_approach_each_processing_propagation_CDs.items():
                for approach,processing_propagation_CDs in each_approach_each_processing_propagation_CDs.items():
                    if approach == each_approach:
                        for processing,propagation_CDs in processing_propagation_CDs.items():
                            if processing == processing_delay:
                                for propagation, CDs in propagation_CDs.items():
                                    if propagation_delay == propagation:
                                        CD_this_approach = sum(CDs)/len(CDs)
                                        try:
                                            zero_batch_CDs = each_event_each_approach_each_processing_propagation_CDs[event]['zero_batching'][processing][propagation]
                                            zero_batch_CD = sum(zero_batch_CDs)/len(zero_batch_CDs)
                                            if CD_this_approach/zero_batch_CD==10.030782029950082:
                                                print('this is unnrmal !!!!!!!',event)
                                            if CD_this_approach/zero_batch_CD<10:
                                                
                                                if CD_this_approach/zero_batch_CD >2:
                                                    print('this is why we do have higher than 2 ',event,CD_this_approach,zero_batch_CD)
                                                list_of_normalized_CDs.append(CD_this_approach/zero_batch_CD)
                                            
                                        except:
                                            pass
            print('this is normalized CDs ',each_approach,int(propagation_delay),list_of_normalized_CDs,sum(list_of_normalized_CDs)/len(list_of_normalized_CDs))
            if list_of_normalized_CDs:
                try:
                    each_approach_each_x_axis_point_values[each_approach][int(propagation_delay)] = list_of_normalized_CDs
                except:
                    #print('error approach, processing propagation',each_approach,processing_delay,int(propagation_delay))
                    each_approach_each_x_axis_point_values[each_approach] = {}
                    each_approach_each_x_axis_point_values[each_approach][int(propagation_delay)] = list_of_normalized_CDs

            #print('each_approach_each_x_axis_point_values at this point ',each_approach_each_x_axis_point_values)
    tickets_on_x_axis = ['[1-1]','[1-50]','[1-100]']
#     tickets_on_x_axis = ['[1-1]','[1-50]']
    #print('each_approach_each_x_axis_point_values',each_approach_each_x_axis_point_values)
   # multiple_box_plot_on_each_x_axis('Propagation delay(ms)','Normalized \n sent messages',tickets_on_x_axis,[1,50,100],each_approach_each_x_axis_point_values,'plots/normalized_CD_propagation_dealy_p=4.pdf')
    
    cdf_info_dictionary_over_multi_item={}
    list_of_keys=[]
    for approach,propagation_delay_CDs in each_approach_each_x_axis_point_values.items():
        if 'Optimal' not in approach:
            for propagation_value ,CDs in propagation_delay_CDs.items():
                approach_key = approach+',[1-'+str(propagation_value)+']'
                list_of_keys.append(approach_key)
                for CD in CDs:
                    try:
                        cdf_info_dictionary_over_multi_item[approach_key][CD] =cdf_info_dictionary_over_multi_item[approach_key][CD] +1
                    except:
                        try:
                            cdf_info_dictionary_over_multi_item[approach_key][CD] = 1
                        except:
                            cdf_info_dictionary_over_multi_item[approach_key]={}
                            cdf_info_dictionary_over_multi_item[approach_key][CD] = 1

    max_value = 0
    min_value=1000
    for approach,propagation_delay_CDs in each_approach_each_x_axis_point_values.items():
        if 'Optimal' not in approach:
            for propagation_value ,CDs in propagation_delay_CDs.items():
                print('******************************* ',max(CDs))
                if min(CDs)<min_value:
                    min_value = min(CDs)
                if  max(CDs)>max_value:
                    max_value =  max(CDs)
    y_min_value= 0.0
    y_max_value = 1.0

    multiple_lines_cdf("Normalized CD regarding zero-batching scheme",'Cumulative fraction \n of events',cdf_info_dictionary_over_multi_item,False,'plots/normalized_CDs_CDF_plot'+str(processing_delay)+'.pdf',list_of_keys,y_min_value,y_max_value)

                    

    




# In[ ]:



tickets_on_x_axis = ['[1-1]','[1-10]']
each_approach_each_x_axis_point_values = {'Work-conserving':{
                                        0:[0.6,0.7,0.7,0.6,0.7,1.4,1.2],
                                        10:[0.6,0.7,0.7,0.6,0.7,1.4,1.2], 
                                        20: [0.7,0.7,0.6,0.7,0.6,0.4,1.1], 
                                        30:[0.8,0.9,0.7,1.2,0.7,1.1]},
                                         'Optimal MRAI':{
                                            0:[0.5,0.5,0.4,0.6,0.5,0.3,0.1,0.1], 
                                             10:[0.5,0.5,0.4,0.6,0.5,0.3,0.1,0.1], 
                                             20: [0.6,0.6,0.4,0.7,0.8,0.9,1.0], 
                                             30:[1.0,0.9,1.0,0.8,0.9,0.7]},
                                         
}
# positions [-0.4  1.6  3.6]


def check_event_in_both_propagations(event,each_approach_each_propagation_processing_delay_MRAI_CD,propagations):
    
    for processing, propagation_CD in each_approach_each_propagation_processing_delay_MRAI_CD[event].items():
        for propagation in propagations:
            if propagation not in propagation_CD.keys():
                return False
    return True
global_each_approach_each_x_axis_point_values = {}
global_each_approach_each_x_axis_point_values_for_optimal_MRAI = {}
list_of_topologies  = set([])
file_result_path = 'each_event_info_result.csv'
with open(file_result_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in (reader):
        
        list_of_topologies.add(line[4])
for topology_size in list(list_of_topologies):
    each_event_ID = {}
    each_approach_each_propagation_processing_delay_MRAI_CD = {} 
    MRAI_values = []
    event_counter = 0
    propagation_values = []
    each_event_each_each_processing_propagation_each_approach_CDs = {}
    processing_delays = []
    propagation_delays = []
    with open(file_result_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in (reader):
            
            if line[4] == topology_size:
                propagation_delay = line[8]
                if propagation_delay not in propagation_delays:
                    propagation_delays.append(propagation_delay)
                procesing_delay= line[7]
                if procesing_delay not in processing_delays:
                    processing_delays.append(procesing_delay)
                MRAI_value = int(line[3])
                if MRAI_value not in MRAI_values:
                    MRAI_values.append((MRAI_value))
                CD = float(line[9])
        #         CD = float(line[10])
                approach = line[1]
                event_title = line[0]
                event_longest_route = line[11]
                optimal_CD = float(line[12])
                use_this_optimal_CD = line[13]
                if  'withdraw' not in event_longest_route and ( 'up' in event_title or 'down' in event_title):
                    if 'Work' in approach or 'work' in approach:
                        approach = 'Work-conserving'
                    if 'MRAI' in approach or 'optimal' in approach:
                        approach = 'MRAI,'+str(MRAI_value)
            #         if 'up' in line[0]:
            #             approach = approach+', UP phase'
            #         else:
            #             approach = approach+', DOWN phase'
                    approach = approach
                
                   
                
                    try:
                        each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach].append(CD)
                    except:
                        try:
                            each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach]= [CD]
                        except:
                            try:
                                each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay] = {}
                                each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach] = [CD]
                            except:
                                try:
                                    each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay] = {}
                                    each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay] = {}
                                    each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach] = [CD]
                                except:
                                    try:
                                        each_event_each_each_processing_propagation_each_approach_CDs[event_title] = {}
                                        each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay] = {}
                                        each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay] = {}
                                        each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach] = [CD]
                                    except ValueError:
                                        print('ValueError',ValueError)
                                        
                                        
                    if use_this_optimal_CD =='yes':
                        approach = 'Physical bound'
                        CD = optimal_CD
                        try:
                            each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach].append(CD)
                        except:
                            try:
                                each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach]= [CD]
                            except:
                                try:
                                    each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay] = {}
                                    each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach] = [CD]
                                except:
                                    try:
                                        each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay] = {}
                                        each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay] = {}
                                        each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach] = [CD]
                                    except:
                                        try:
                                            each_event_each_each_processing_propagation_each_approach_CDs[event_title] = {}
                                            each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay] = {}
                                            each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay] = {}
                                            each_event_each_each_processing_propagation_each_approach_CDs[event_title][procesing_delay][propagation_delay][approach] = [CD]
                                        except ValueError:
                                            print('ValueError',ValueError)
    minimum_mrai_value = min(MRAI_values)
    zero_batching_key = 'MRAI,'+str(minimum_mrai_value)
    apporaches = ['Optimal MRAI','Work-conserving']


    each_approach_processing_each_propagation_convergence_delay_reduction = {}
#     print('MRAI_values',MRAI_values)

    new_x_axis_tickets = []

            #ploting_simple_y_as_x('MRAI(msec) d='+str(each_propagation_delay),'CD(ms) ',processing_delay_as_schemes_in_order,each_processing_scheme_MRAI_CD_value,MRAI_value_for_x_axis,MRAI_value_for_x_axis,False,'plots/CD_focus_MRAI_processing_delay_propagation_'+str(each_propagation_delay)+'.pdf')
    print('done')
    approaches = ['Optimal MRAI','Work-conserving','Physical bound']
    approaches = ['Work-conserving','Optimal MRAI']
#     approaches = ['Optimal MRAI','Optimal']
#     print('processing_delays',processing_delays)
#     print('propagation_delays',propagation_delays)
    events_that_did_not_have_both_schemes = set([])
#     print(' each_event_each_each_processing_propagation_each_approach_CDs :',each_event_each_each_processing_propagation_each_approach_CDs)
    for processing_delay in processing_delays:
        each_approach_each_x_axis_point_values = {}
        for propagation_delay_value in propagation_delays:
            for approach in approaches:
                list_of_normalized_CDs = []
                list_of_optimal_MRAI_values = []
                if approach =='Work-conserving':        
                    for each_event,each_each_processing_propagation_each_approach_CDs in each_event_each_each_processing_propagation_each_approach_CDs.items():
                        for each_processing_delay, propagation_each_approach_CDs in each_each_processing_propagation_each_approach_CDs.items():
                            if processing_delay ==each_processing_delay:
                                for each_propagation,each_approach_CDs in propagation_each_approach_CDs.items():
                                    if each_propagation ==propagation_delay_value:
                                        try:
                                            work_conserving_CDs = each_event_each_each_processing_propagation_each_approach_CDs[each_event][each_processing_delay][each_propagation][approach]
                                            zero_batching_CDs = each_event_each_each_processing_propagation_each_approach_CDs[each_event][each_processing_delay][each_propagation][zero_batching_key]
                                            mean_CD_work_conserving = sum(work_conserving_CDs)/len(work_conserving_CDs)
                                            mean_CD_zero_batching = sum(zero_batching_CDs)/len(zero_batching_CDs)
                                            noralized_CD = mean_CD_work_conserving/mean_CD_zero_batching
                                            #f noralized_CD <2:
                                            list_of_normalized_CDs.append(noralized_CD)
                                        except:
                                            events_that_did_not_have_both_schemes.add(each_event)


                elif approach =='Optimal MRAI':


                    for each_event,each_each_processing_propagation_each_approach_CDs in each_event_each_each_processing_propagation_each_approach_CDs.items():
                        all_MRAI_normalized_CD =[]
                        minimum_CD_MRAI = {}
                        for each_processing_delay, propagation_each_approach_CDs in each_each_processing_propagation_each_approach_CDs.items():
                            if processing_delay ==each_processing_delay:
                                for each_propagation,each_approach_CDs in propagation_each_approach_CDs.items():
                                    if each_propagation ==propagation_delay_value:
                                        for each_scheme,CDs in each_approach_CDs.items():
                                            if each_scheme !='Work-conserving' and each_scheme!='Physical bound':
                                                try:
                                                    this_MRAI_CDs = each_event_each_each_processing_propagation_each_approach_CDs[each_event][each_processing_delay][each_propagation][each_scheme]
                                                    zero_batching_CDs = each_event_each_each_processing_propagation_each_approach_CDs[each_event][each_processing_delay][each_propagation][zero_batching_key]
                                                    mean_CD_this_MRAI = sum(this_MRAI_CDs)/len(this_MRAI_CDs)
                                                    mean_CD_zero_batching = sum(zero_batching_CDs)/len(zero_batching_CDs)
                                                    noralized_CD = mean_CD_this_MRAI/mean_CD_zero_batching
                                                    
                                                    all_MRAI_normalized_CD.append(noralized_CD)
                                                    MRAI_value_scheme = int(each_scheme.split(",")[1])
                                                    minimum_CD_MRAI[noralized_CD] = MRAI_value_scheme
                                                except:
                                                    events_that_did_not_have_both_schemes.add(each_event)

                        if all_MRAI_normalized_CD:
                            #f min(all_MRAI_normalized_CD)<2:
#                             print('error',all_MRAI_normalized_CD)
#                             print('************************ we are selecing min %s among %s'%(min(all_MRAI_normalized_CD),(all_MRAI_normalized_CD)))
#                             print('minimum_CD_MRAI',minimum_CD_MRAI)
                            optimal_MRAI_value_for_this_event = minimum_CD_MRAI[min(all_MRAI_normalized_CD)]
                            list_of_optimal_MRAI_values.append(optimal_MRAI_value_for_this_event)
                            list_of_normalized_CDs.append(min(all_MRAI_normalized_CD))

                elif approach =='Physical bound':
                    for each_event,each_each_processing_propagation_each_approach_CDs in each_event_each_each_processing_propagation_each_approach_CDs.items():
                        for each_processing_delay, propagation_each_approach_CDs in each_each_processing_propagation_each_approach_CDs.items():
                            if processing_delay ==each_processing_delay:
                                for each_propagation,each_approach_CDs in propagation_each_approach_CDs.items():
                                    if each_propagation ==propagation_delay_value:
                                        try:
                                            optimal_scheme_CDs = each_event_each_each_processing_propagation_each_approach_CDs[each_event][each_processing_delay][each_propagation][approach]
                                            zero_batching_CDs = each_event_each_each_processing_propagation_each_approach_CDs[each_event][each_processing_delay][each_propagation][zero_batching_key]
                                            mean_CD_optimal_scheme = sum(optimal_scheme_CDs)/len(optimal_scheme_CDs)
                                            mean_CD_zero_batching = sum(zero_batching_CDs)/len(zero_batching_CDs)
                                            noralized_CD = mean_CD_optimal_scheme/mean_CD_zero_batching
                                            #if noralized_CD <2:
                                            list_of_normalized_CDs.append(noralized_CD)
                                        except:
                                            events_that_did_not_have_both_schemes.add(each_event)

                if list_of_normalized_CDs:
                    try:
                        each_approach_each_x_axis_point_values[approach][int(propagation_delay_value)] = list_of_normalized_CDs
                    except:
                        #print('error approach, processing propagation',each_approach,processing_delay,int(propagation_delay))
                        each_approach_each_x_axis_point_values[approach] = {}
                        each_approach_each_x_axis_point_values[approach][int(propagation_delay_value)] = list_of_normalized_CDs
                    print('for scheme %s propagation delay %s we are adding to the global x_pints-Values'%(approach,propagation_delay_value))
                    try:
                        global_each_approach_each_x_axis_point_values[approach][processing_delay][int(propagation_delay_value)].extend(list_of_normalized_CDs)
                    except:
                        try:
                            global_each_approach_each_x_axis_point_values[approach][processing_delay][int(propagation_delay_value)] = list_of_normalized_CDs
                        except:
                            try:
                                global_each_approach_each_x_axis_point_values[approach][processing_delay] = {}
                                global_each_approach_each_x_axis_point_values[approach][processing_delay][int(propagation_delay_value)] = list_of_normalized_CDs
                            except:
                                global_each_approach_each_x_axis_point_values[approach] = {}
                                global_each_approach_each_x_axis_point_values[approach][processing_delay] = {}
                                global_each_approach_each_x_axis_point_values[approach][processing_delay][int(propagation_delay_value)] = list_of_normalized_CDs
                                
                    try:
                        global_each_approach_each_x_axis_point_values_for_optimal_MRAI[approach][int(propagation_delay_value)].extend(list_of_optimal_MRAI_values)
                    except:
                        global_each_approach_each_x_axis_point_values_for_optimal_MRAI[approach] = {}
                        global_each_approach_each_x_axis_point_values_for_optimal_MRAI[approach][int(propagation_delay_value)] = list_of_optimal_MRAI_values
        tickets_on_x_axis = ['[1-1]','[1-50]','[1-100]']

        cdf_info_dictionary_over_multi_item={}
        list_of_keys=[]
        for approach,propagation_delay_CDs in each_approach_each_x_axis_point_values.items():
            for propagation_value ,CDs in propagation_delay_CDs.items():
                approach_key = approach+',[1-'+str(propagation_value)+']'
                list_of_keys.append(approach_key)
                for CD in CDs:
                    try:
                        cdf_info_dictionary_over_multi_item[approach_key][CD] =cdf_info_dictionary_over_multi_item[approach_key][CD] +1
                    except:
                        try:
                            cdf_info_dictionary_over_multi_item[approach_key][CD] = 1
                        except:
                            cdf_info_dictionary_over_multi_item[approach_key]={}
                            cdf_info_dictionary_over_multi_item[approach_key][CD] = 1

    #     max_value = 0
    #     min_value=1000
    #     for approach,propagation_delay_CDs in each_approach_each_x_axis_point_values.items():

    #         for propagation_value ,CDs in propagation_delay_CDs.items():
    #             print('******************************* ',max(CDs))
    #             if min(CDs)<min_value:
    #                 min_value = min(CDs)
    #             if  max(CDs)>max_value:
    #                 max_value =  max(CDs)
        y_min_value= 0.0
        y_max_value = 1.0
        #print('these are the numbers we have ',cdf_info_dictionary_over_multi_item)
        
        
        for s, propagation_points in each_approach_each_x_axis_point_values.items():
            for d,v in propagation_points.items():
                print('******* scheme %s for propagation delay value %s has %s points ********'%(s,d,len(v)))
            
        multiple_lines_cdf("Normalized CD for topology "+topology_size+', p='+str(processing_delay),'Cumulative fraction \n of events',cdf_info_dictionary_over_multi_item,False,'plots/normalized_CDs_CDF_plot'+str(processing_delay)+'_'+str(topology_size)+'.pdf',list_of_keys,y_min_value,y_max_value)







#     print(len(list(events_that_did_not_have_both_schemes)),events_that_did_not_have_both_schemes)
    
each_event_each_each_processing_propagation_each_approach_CDs
print('global_each_approach_each_x_axis_point_values',global_each_approach_each_x_axis_point_values)

for processing_delay in processing_delays:
    cdf_info_dictionary_over_multi_item={}
    list_of_keys=[]
    for approach,processing_propagation_delay_CDs in global_each_approach_each_x_axis_point_values.items():
        for each_p, propagation_delay_CDs in processing_propagation_delay_CDs.items():
            if each_p == processing_delay:
                for propagation_value ,CDs in propagation_delay_CDs.items():
                    approach_key = approach+',[1-'+str(propagation_value)+']'
                    list_of_keys.append(approach_key)
                    for CD in CDs:
                        try:
                            cdf_info_dictionary_over_multi_item[approach_key][CD] =cdf_info_dictionary_over_multi_item[approach_key][CD] +1
                        except:
                            try:
                                cdf_info_dictionary_over_multi_item[approach_key][CD] = 1
                            except:
                                cdf_info_dictionary_over_multi_item[approach_key]={}
                                cdf_info_dictionary_over_multi_item[approach_key][CD] = 1
    y_min_value= 0.0
    y_max_value = 1.0
    print('list_of_keys for global one ',list_of_keys)
    multiple_lines_cdf("Normalized CD",'Cumulative fraction \n of events',cdf_info_dictionary_over_multi_item,False,'plots/global_normalized_CDs_CDF_plot_p_'+str(processing_delay)+'.pdf',list_of_keys,y_min_value,y_max_value)


# In[ ]:





# In[ ]:


cdf_info_dictionary_over_multi_item={}
list_of_keys=[]
for approach,propagation_delay_CDs in global_each_approach_each_x_axis_point_values_for_optimal_MRAI.items():
    for propagation_value ,CDs in propagation_delay_CDs.items():
        approach_key = approach+',[1-'+str(propagation_value)+']'
        list_of_keys.append(approach_key)
        for CD in CDs:
            try:
                cdf_info_dictionary_over_multi_item[approach_key][CD] =cdf_info_dictionary_over_multi_item[approach_key][CD] +1
            except:
                try:
                    cdf_info_dictionary_over_multi_item[approach_key][CD] = 1
                except:
                    cdf_info_dictionary_over_multi_item[approach_key]={}
                    cdf_info_dictionary_over_multi_item[approach_key][CD] = 1
y_min_value= 0.0
y_max_value = 1.0

multiple_lines_cdf("Optimal MRAI (ms)",'Cumulative fraction \n of events',cdf_info_dictionary_over_multi_item,False,'plots/global_optimal_MRAI_CDF_plot'+str(processing_delay)+'.pdf',list_of_keys,y_min_value,y_max_value)


# In[ ]:


print("plotting normalized CD for synthetic topologies")
#python Parsing_synthetic_topologies_results_per_MRAI_processing_propagation_delay.py  [] [180] [optimal_MRAI] [0,10,15] [1,50,100] [100] yes delete yes no
# synthetic_experiment_work_conserving_scheme_focus_topology_final_version
# synthetic_experiment_optimal_MRAI_focus_topology
# synthetic_experiment_optimal_MRAI_focus_topology_random_propagation_delay
file_result_path = 'each_event_info_result.csv'
target_MRAI_as_zero_batching = '0'
each_approach_each_propagation_processing_delay_MRAI_CD = {} 
MRAI_values = []
with open(file_result_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in (reader):
        propagation_delay = line[8]
        procesing_delay= line[7]
        MRAI_value = int(line[3])
        if MRAI_value not in MRAI_values:
            MRAI_values.append((MRAI_value))
        CD = float(line[9])
#         CD = float(line[10])
        approach = line[1]
        if 'up' in line[0]:
            if 'Work' in approach or 'work' in approach:
                approach = 'Work-conserving'
            if 'optimal_MRAI' in approach:
                approach = 'Optimal MRAI'
            if 'MRAI_scheme' in approach:
                approach = '5 sec. MRAI'
    #         if 'up' in line[0]:
    #             approach = approach+', UP phase'
    #         else:
    #             approach = approach+', DOWN phase'
            approach = approach
            try:
                each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value].append(CD)
            except:
                try:
                    each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                except:
                    try:
                        each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                        each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                    except:
                        try:
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                        except:
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay]={}
                            each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][procesing_delay][MRAI_value]=[CD]
                            
for approach, each_propagation_processing_delay_MRAI_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
    for propagation_delay,processing_delay_MRAI_CDs in each_propagation_processing_delay_MRAI_CD.items():
        for processing_delay,MRAI_CDs in processing_delay_MRAI_CDs.items():
            for MRAI,CDs in MRAI_CDs.items():
                each_approach_each_propagation_processing_delay_MRAI_CD[approach][propagation_delay][processing_delay][MRAI] = sum(CDs)/len(CDs)


# processing_delays = ['4','20','50']
# propagation_delays = [1]
print('each_approach_each_propagation_processing_delay_MRAI_CD is ',each_approach_each_propagation_processing_delay_MRAI_CD)
each_approach_processing_each_propagation_convergence_delay_reduction = {}
print('MRAI_values',MRAI_values)
propagation_delays = []
new_x_axis_tickets = []
for approach, each_propagation_processing_delay_MRAI_CD in each_approach_each_propagation_processing_delay_MRAI_CD.items():
    for each_propagation_delay,each_processing_delay_MRAI_CD in each_propagation_processing_delay_MRAI_CD.items():
        processing_delay_as_schemes_in_order = []
        processing_delays = list(each_processing_delay_MRAI_CD.keys())
        if int(each_propagation_delay) not in propagation_delays:
            propagation_delays.append(int(each_propagation_delay))
        if '[1-'+str(each_propagation_delay)+']' not in new_x_axis_tickets:
            new_x_axis_tickets.append('[1-'+str(each_propagation_delay)+']')
        schemes_in_order = processing_delays
    #     each_processing_delay_MRAI_CD = {'4':{0:4,5:2,10:6},
    #                                     '20':{0:4,5:2,10:6},
    #                                     '50':{0:4,5:2,10:6},}
        MRAI_values.sort()
        print('each_processing_delay_MRAI_CD ',each_processing_delay_MRAI_CD)
        print('MRAI_values ',MRAI_values)

        #tickets_on_x_axis = ['[1-10]','[10-50]','[50-100]']
        tickets_on_x_axis =MRAI_values
        each_processing_scheme_MRAI_CD_value = {}
        MRAI_value_for_x_axis = []
        # positions [-0.4  1.6  3.6]
        #multiple_box_plot_on_each_x_axis('Propagation delay(ms)','Average queue size',tickets_on_x_axis,[0,10,20],each_approach_each_x_axis_pont_values,'plots/queuing_size_propagation_dealy_p=5.pdf')
        for processing_delay,MRAI_CD in each_processing_delay_MRAI_CD.items():
            if 'UP' in approach:
                zero_batch_CD = each_approach_each_propagation_processing_delay_MRAI_CD['Optimal MRAI'+', UP phase'][each_propagation_delay][processing_delay][0]
            else:
                zero_batch_CD = each_approach_each_propagation_processing_delay_MRAI_CD['Optimal MRAI'][each_propagation_delay][processing_delay][0]  
            convergence_delays = []
            if 'p='+str(processing_delay) not in processing_delay_as_schemes_in_order:
                processing_delay_as_schemes_in_order.append('p='+str(processing_delay))
            for MRAI, CD in MRAI_CD.items():
                if MRAI not in MRAI_value_for_x_axis:
                    MRAI_value_for_x_axis.append(MRAI)
                if 'Work-conserving' in approach:
                    if int(MRAI)==1000:
                        convergence_delays.append(CD)
                else:
                    if int(MRAI)!=1000:
                        convergence_delays.append(CD)
                if 'Optimal MRAI' in approach:
                    if int(MRAI)!=1000:
                        try:
                            each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD
                        except:
                            each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)]={}
                            each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD
                else:
                    try:
                        each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD
                    except:
                        each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)]={}
                        each_processing_scheme_MRAI_CD_value['p='+str(processing_delay)][MRAI] = CD   
                    
            convergence_delay_reduction = (zero_batch_CD-min(convergence_delays))/zero_batch_CD
            convergence_delay_reduction =min(convergence_delays)/zero_batch_CD
            if convergence_delay_reduction >1:
                print('these should not happen min()CDs %s zero CD %s scheme %s '%(min(convergence_delays),zero_batch_CD,approach))
            try:
                each_approach_processing_each_propagation_convergence_delay_reduction[approach+', p='+processing_delay+'\u03BCs'][int(each_propagation_delay)] = convergence_delay_reduction
            except:
                each_approach_processing_each_propagation_convergence_delay_reduction[approach+', p='+processing_delay+'\u03BCs'] = {}
                each_approach_processing_each_propagation_convergence_delay_reduction[approach+', p='+processing_delay+'\u03BCs'][int(each_propagation_delay)] = convergence_delay_reduction
        MRAI_value_for_x_axis.sort()
        #ploting_simple_y_as_x('MRAI(msec) d='+str(each_propagation_delay),'CD(ms) ',processing_delay_as_schemes_in_order,each_processing_scheme_MRAI_CD_value,MRAI_value_for_x_axis,MRAI_value_for_x_axis,False,'plots/CD_focus_MRAI_processing_delay_propagation_'+str(each_propagation_delay)+'.pdf')
print('done')
schemes_in_order = []
processing_delays.sort()
for approach in each_approach_each_propagation_processing_delay_MRAI_CD:
    for processing_delay in processing_delays:
        schemes_in_order.append(approach+', p='+str(processing_delay)+'\u03BCs')
print('each_processing_each_propagation_convergence_delay_reduction ',each_approach_processing_each_propagation_convergence_delay_reduction)
print('schemes_in_order ',schemes_in_order)
print('propagation_delays ',propagation_delays)
propagation_delays.sort()
print('propagation_delays ',propagation_delays)
new_x_axis_tickets = []
for sorted_x_axis_value in propagation_delays:
    new_x_axis_tickets.append('[1-'+str(sorted_x_axis_value)+']')
print('new_x_axis_tickets ',new_x_axis_tickets)
plot_convergence_detection_alg_overhead('Propagation delay range(msec): log scale','Normalized CD',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,new_x_axis_tickets,False,'plots/normalized_CD_focus_propagation_delay.pdf')

ploting_simple_y_as_x(                  'Propagation delay(msec)','Normalized CD',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,new_x_axis_tickets,False,'plots/normalized_CD_focus_propagation_delay_no_log.pdf')




# In[ ]:


normalized_CD_for_each_propagation_range={
                                    'Work-conserving':{'1':10,'20':14,'50':20,'100':20,'200':22,'400':23},
                                    'Optimal MRAI':{   '1' :10,'20':16,'50':20,'100':25,'200':34,'400':35},
                                    '5-sec. MRAI timer':{'1':10,'20':16,'50':20,'100':25,'200':34,'400':35}
                                        }
x_axis_labels = ['1','20','50','100','200','400']



print(propagation_delays,each_approach_processing_each_propagation_convergence_delay_reduction)


# plot_bar_plot('Propagation delay(ms)','Normalized CD',normalized_CD_for_each_propagation_range,x_axis_labels,'plots/bar_plot_normalized_CD_WC_MRAI_schemes.pdf')
plot_bar_plot('Propagation delay(ms): log scale','Normalized CD',each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,'plots/bar_plot_normalized_CD_WC_MRAI_schemes_real.pdf')

#              (x_axis_label,          y_axis_label,    invalid_routes_percentage_based_on_rho_compression,x_axis_labels,plot_file_name)


# In[ ]:


if 929.5>929.5:
    print('yes')


# In[ ]:


schemes_in_order = ['MRAI scheme','Work-conserving']
each_approach_processing_each_propagation_convergence_delay_reduction = {'MRAI scheme':{4:6,20:7,50:10,100:14,500:17},
                                                                       'Work-conserving':{4:8,20:9,50:12,100:18,500:21} }
propagation_delays = [4,20,50,100,500]

new_x_axis_tickets = [4,20,50,100,500]

each_approach_processing_each_propagation_convergence_delay_reductions = {}
schemes_in_order = []
MRAI_values = []
file_result_path = 'each_event_info_result.csv'
each_topology_k = {'150':'50','30':'10'}
with open(file_result_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in (reader):    
            MRAI_value = int(line[3])
            if MRAI_value not in MRAI_values:
                MRAI_values.append((MRAI_value))
            CD = float(line[9])/1000
            #CD = float(line[10])
            approach = line[1]
            event_title = line[0]
            event_longest_route = line[11]
            optimal_CD = float(line[12])
            use_this_optimal_CD = line[13]
            procesing_delay= int(line[7])
            k= each_topology_k[line[4]]
            print(MRAI_value)
            if  'withdraw' not in event_longest_route and ('up' in event_title):
                if 'Work' in approach or 'work' in approach:
                    approach = 'Infinite work-conserving,k='+str(k)
                if 'MRAI' in approach or 'optimal' in approach:
                    approach = 'Zero batching non-work-conserving(MRAI=0 sec),k='+str(k)
                if approach not in schemes_in_order:
                    schemes_in_order.append(approach)
                try: 
                    each_approach_processing_each_propagation_convergence_delay_reductions[approach][procesing_delay].append(CD)
                except:
                    try:
                        each_approach_processing_each_propagation_convergence_delay_reductions[approach][procesing_delay]=[CD]
                    except:
                        each_approach_processing_each_propagation_convergence_delay_reductions[approach]={}
                        each_approach_processing_each_propagation_convergence_delay_reductions[approach][procesing_delay]= [CD]

print('each_approach_processing_each_propagation_convergence_delay_reductions',each_approach_processing_each_propagation_convergence_delay_reductions)
each_approach_processing_each_propagation_convergence_delay_reduction = {}
propagation_delays = []
for scheme,p_CDs in each_approach_processing_each_propagation_convergence_delay_reductions.items():
    for p,CDs in p_CDs.items():
        CD = sum(CDs)/len(CDs)
        if p not in propagation_delays:
            propagation_delays.append(int(p))
        try:
            each_approach_processing_each_propagation_convergence_delay_reduction[scheme][p] = CD
        except:
            each_approach_processing_each_propagation_convergence_delay_reduction[scheme] = {}
            each_approach_processing_each_propagation_convergence_delay_reduction[scheme][p] = CD
new_x_axis_tickets = propagation_delays
print('propagation_delays ',propagation_delays)
propagation_delays.sort()
print('propagation_delays ',propagation_delays)
new_x_axis_tickets = []
for sorted_x_axis_value in propagation_delays:
    new_x_axis_tickets.append(str(sorted_x_axis_value))  
    

plot_convergence_detection_alg_overhead('Processing delay(ms)','CD(sec)',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,new_x_axis_tickets,False,'plots/batch_for_no_good_CD_focus_topology.pdf')



# In[ ]:


each_approach_processing_each_propagation_convergence_delay_reductions = {}
schemes_in_order = []
MRAI_values = []
file_result_path = 'each_event_info_result.csv'
with open(file_result_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in (reader):    
            MRAI_value = int(line[3])
            if MRAI_value not in MRAI_values:
                MRAI_values.append((MRAI_value))
            CD = float(line[9])/1000
#             CD = float(line[10])
            approach = line[1]
            event_title = line[0]
            event_longest_route = line[11]
            optimal_CD = float(line[12])
            use_this_optimal_CD = line[13]
            procesing_delay= line[7]
            propagation_delay =  line[8]
            print(MRAI_value)
            if  'withdraw' not in event_longest_route and ('up' in event_title):
                if 'Work' in approach or 'work' in approach:
                    approach = 'Work-conserving'
                if 'MRAI' in approach or 'optimal' in approach:
                    approach = 'MRAI scheme,'+propagation_delay
                if approach not in schemes_in_order:
                    schemes_in_order.append(approach)
                try: 
                    each_approach_processing_each_propagation_convergence_delay_reductions[approach][MRAI_value].append(CD)
                except:
                    try:
                        each_approach_processing_each_propagation_convergence_delay_reductions[approach][MRAI_value]=[CD]
                    except:
                        each_approach_processing_each_propagation_convergence_delay_reductions[approach]={}
                        each_approach_processing_each_propagation_convergence_delay_reductions[approach][MRAI_value]= [CD]

                            
#schemes_in_order = ['MRAI scheme','Infinite work-conserving']
# each_approach_processing_each_propagation_convergence_delay_reduction ={'MRAI scheme':{0:14,100:12,200:10,400:8,600:8,800:10,1000:12,1200:14,1400:16,1600:18},
# 'Infinite work-conserving':{0:12,100:12,200:12,400:12,600:12,800:12,1000:12,1200:12,1400:12,1600:12} }
print('each_approach_processing_each_propagation_convergence_delay_reductions',each_approach_processing_each_propagation_convergence_delay_reductions)
each_approach_processing_each_propagation_convergence_delay_reduction = {}
propagation_delays = []
for scheme,MRAI_CDs in each_approach_processing_each_propagation_convergence_delay_reductions.items():
    if 'MRAI scheme' in scheme:
        for MRAI,CDs in MRAI_CDs.items():
            CD = sum(CDs)/len(CDs)
            if MRAI not in propagation_delays:
                propagation_delays.append(int(MRAI))
            try:
                each_approach_processing_each_propagation_convergence_delay_reduction[scheme][MRAI] = CD
            except:
                each_approach_processing_each_propagation_convergence_delay_reduction[scheme] = {}
                each_approach_processing_each_propagation_convergence_delay_reduction[scheme][MRAI] = CD
for scheme,MRAI_CDs in each_approach_processing_each_propagation_convergence_delay_reductions.items():
    if 'Work-conserving' in scheme:
        for MRAI,CDs in MRAI_CDs.items():
            CD = sum(CDs)/len(CDs)
            for mrai in propagation_delays:
                try:
                    each_approach_processing_each_propagation_convergence_delay_reduction[scheme][mrai] = CD
                except:
                    each_approach_processing_each_propagation_convergence_delay_reduction[scheme] = {}
                    each_approach_processing_each_propagation_convergence_delay_reduction[scheme][mrai] = CD   
            break

print('each_approach_processing_each_propagation_convergence_delay_reduction',each_approach_processing_each_propagation_convergence_delay_reduction)
new_x_axis_tickets = propagation_delays
print('propagation_delays ',propagation_delays)
propagation_delays.sort()
print('propagation_delays ',propagation_delays)
new_x_axis_tickets = []
for sorted_x_axis_value in propagation_delays:
    new_x_axis_tickets.append(str(sorted_x_axis_value))
    
# schemes_in_order = ['MRAI scheme','Work-conserving']
# propagation_delays = [0,100,200,400,600,800,1000,1200,1400,1600]

# new_x_axis_tickets = [0,100,200,300,400,500,600,700,800,1000,1200,1400,1600]
# each_approach_processing_each_propagation_convergence_delay_reduction ={'MRAI scheme':{0:14,100:12,200:10,400:8,600:8,800:10,1000:12,1200:14,1400:16,1600:18},
# 'Work-conserving':{0:12,100:12,200:12,400:12,600:12,800:12,1000:12,1200:12,1400:12,1600:12} }
# plot_convergence_detection_alg_overhead('MRAI(ms)','CD(sec)',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,new_x_axis_tickets,False,'plots/not_able_to_reduce_MC_CD_focus_topology.pdf')
plot_convergence_detection_alg_overhead('Parameter/MRAI','CD(sec)',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,new_x_axis_tickets,False,'plots/not_able_to_reduce_MC_CD_focus_topology.pdf')



# In[ ]:


each_approach_processing_each_propagation_convergence_delay_reductions = {}
schemes_in_order = []
MRAI_values = []
each_topology_k = {'150':'50','30':'10','156':'50','56':'20','106':'40','50':'50','100':'100','10':'10'}
file_result_path = 'each_event_info_result.csv'
with open(file_result_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in (reader):    
        MRAI_value = int(line[3])
        if MRAI_value ==1000:
            MRAI_value = 100
        if MRAI_value not in MRAI_values:
            MRAI_values.append((MRAI_value))
        CD = float(line[9])/1000
#             CD = float(line[10])
        approach = line[1]
        event_title = line[0]
        event_longest_route = line[11]
        optimal_CD = float(line[12])
        use_this_optimal_CD = line[13]
        procesing_delay= line[7]
        propagation_delay =  line[8]
        k= each_topology_k[line[4]]

        print(MRAI_value)
        if  'withdraw' not in event_longest_route and ('up' in event_title or 'down' in event_title):
            if 'Work' in approach or 'work' in approach:
                approach = 'k='+str(k)

            if approach not in schemes_in_order:
                schemes_in_order.append(approach)
            try: 
                each_approach_processing_each_propagation_convergence_delay_reductions[approach][MRAI_value].append(CD)
            except:
                try:
                    each_approach_processing_each_propagation_convergence_delay_reductions[approach][MRAI_value]=[CD]
                except:
                    each_approach_processing_each_propagation_convergence_delay_reductions[approach]={}
                    each_approach_processing_each_propagation_convergence_delay_reductions[approach][MRAI_value]= [CD]

                            
#schemes_in_order = ['MRAI scheme','Infinite work-conserving']
# each_approach_processing_each_propagation_convergence_delay_reduction ={'MRAI scheme':{0:14,100:12,200:10,400:8,600:8,800:10,1000:12,1200:14,1400:16,1600:18},
# 'Infinite work-conserving':{0:12,100:12,200:12,400:12,600:12,800:12,1000:12,1200:12,1400:12,1600:12} }
print('each_approach_processing_each_propagation_convergence_delay_reductions',each_approach_processing_each_propagation_convergence_delay_reductions)
each_approach_processing_each_propagation_convergence_delay_reduction = {}
print('each_approach_processing_each_propagation_convergence_delay_reduction',each_approach_processing_each_propagation_convergence_delay_reduction)
propagation_delays = []
new_x_axis_tickets = []
for scheme,parameter_CDs in each_approach_processing_each_propagation_convergence_delay_reductions.items():
    for parameter,CDs in parameter_CDs.items():
        CD = sum(CDs)/len(CDs)
        base_scheme_CDs = each_approach_processing_each_propagation_convergence_delay_reductions[scheme][1]
        base_scheme_CD = sum(base_scheme_CDs)/len(base_scheme_CDs)
        normalized_CD = CD/base_scheme_CD
        print('normalized_CD is ',CD,base_scheme_CD,normalized_CD)
        if int(parameter) not in new_x_axis_tickets:
            new_x_axis_tickets.append(int(parameter))
        try:
            each_approach_processing_each_propagation_convergence_delay_reduction[scheme][parameter] = normalized_CD
        except:
            each_approach_processing_each_propagation_convergence_delay_reduction[scheme] = {}
            each_approach_processing_each_propagation_convergence_delay_reduction[scheme][parameter] = normalized_CD   
        

print('each_approach_processing_each_propagation_convergence_delay_reduction',each_approach_processing_each_propagation_convergence_delay_reduction)
parameter_values = new_x_axis_tickets
print('parameter_values ',parameter_values)
parameter_values.sort()
print('parameter_values ',parameter_values)

# for sorted_x_axis_value in propagation_delays:
#     new_x_axis_tickets.append(str(sorted_x_axis_value))
    
# schemes_in_order = ['MRAI scheme','Work-conserving']
# propagation_delays = [0,100,200,400,600,800,1000,1200,1400,1600]

# new_x_axis_tickets = [0,100,200,300,400,500,600,700,800,1000,1200,1400,1600]
# each_approach_processing_each_propagation_convergence_delay_reduction ={'MRAI scheme':{0:14,100:12,200:10,400:8,600:8,800:10,1000:12,1200:14,1400:16,1600:18},
# 'Work-conserving':{0:12,100:12,200:12,400:12,600:12,800:12,1000:12,1200:12,1400:12,1600:12} }
# plot_convergence_detection_alg_overhead('MRAI(ms)','CD(sec)',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,propagation_delays,new_x_axis_tickets,False,'plots/not_able_to_reduce_MC_CD_focus_topology.pdf')
plot_convergence_detection_alg_overhead('Work-conserving parameter (b)','Normalized CD',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,parameter_values,new_x_axis_tickets,False,'plots/WC_parameter_normalized_CD_focus_topology.pdf')
ploting_simple_y_as_x('Work-conserving parameter (b)','Normalized CD',schemes_in_order,each_approach_processing_each_propagation_convergence_delay_reduction,parameter_values,new_x_axis_tickets,False,'plots/WC_parameter_normalized_CD_focus_topology_no_log.pdf')


# In[ ]:



each_scheme_parameter_CD_value= {'k=50':{1:1,2:1,3:0.9,5:0.8,8:0.7,10:0.7,15:0.5,20:0.4,25:0.3,30:0.4,40:0.6,50:0.7,60:0.7},
                        'k=20':{1:1,2:0.8,3:0.6,5:0.4,8:0.5,10:0.6,15:0.6,20:0.6,25:0.6,30:0.6,40:0.6,50:0.6,60:0.6}}
k_values_as_schemes = ['k=20','k=50']
values_for_x_axis = [1,2,3,5,8,10,15,20,25,30,40,50,60]

ploting_simple_y_as_x('Work-conserving parameter (b)','Normalized CD',k_values_as_schemes,each_scheme_parameter_CD_value,values_for_x_axis,values_for_x_axis,False,'plots/normalized_CD_focus_parameterized_WC.pdf')


# In[ ]:



# alpha = 0 
# r2 1 r3 1 weighted_sum 0.632 C2 1.421 C3 0.632 TDM 0.889
# r2 0 r3 0 weighted_sum 0.867 C2 1.385 C3 0.633 TDM 0.869
# alpha = 0.2 
# r2 1 r3 0.47 weighted_sum 0.276 C2 1.381 C3 0.635 TDM 0.87
# alpha= 0.4
# r2 0.085 r3 0.123 weighted_sum 0.382 C2 1.384 C3 0.635 TDM 0.869
#alpja =0.6
# r2 1 r3 0.88 weighted_sum 0.829 C2 1.382 C3 0.634 TDM 0.868
#alpha = 0.8
# r2 0.445 r3 0.397 weighted_sum 0.826 C2 1.380 C3 0.632 TDM 0.869

# alpha = 1
# r2 1 r3 0.113 weighted_sum 1.376 C2 1.376 C3 0.632 TDM 0.864
each_k_alpha_capacity= {'1 src-dst':{0.0:1.0,10:0.9,20:0.89,30:0.829,40:0.726,50.0:0.70},
                       '2 src-dst':{0.0:0.869,0.2:0.869,0.4:0.869,0.6:0.869,0.8:0.869,1.0:0.869},
                        '3 src-dst':{0.0:1.382,0.2:1.382,0.4:1.382,0.6:1.382,0.8:1.382,1.0:1.382},
                        'C3':{0.0:0.632,0.2:0.632,0.4:0.632,0.6:0.632,0.8:0.632,1.0:0.632},
                       }
adapting_schemes = ['1 src-dst']
values_for_x_axis = [0,10,20,30,40,50]
values_for_x_axis.sort()
ploting_simple_y_as_x('Workload range','Average Fidelity',adapting_schemes,each_k_alpha_capacity,values_for_x_axis,values_for_x_axis,False,'plots/average_fidelity.pdf')


# In[ ]:





# 
