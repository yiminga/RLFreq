# -*- coding:UTF-8 -*-
'''
wym 2020.11.28
pkg power capping + ED2P
'''
import numpy as np
import random
import os
'''
reward:		power/energy-pkg/ 
State space:
0	insn per cycle	=  instructions/ cpu-cycles
1   MPO
'''
FREQ_IN_STATE=0
cpu_last_action = None
cpu_last_state = None
dram_last_action = None
dram_last_state = None
uf_last_action = None
uf_last_state = None
EPSILON = 0.3        # greedy police  train：0.3  test：0.2
GAMMA = 0.8            # Discounting factor  0.95
ALPHA = 0.7           # learning rate  0.5~0.9 0.7-> bad   0.1
EPSILON1 = 0.5   # update Q_A OR Q_B
'''
q-learning  init  runtime vars
'''
CPU_LABELS = ('IPC',
              'MPO'
              )
UF_LABELS = ('IPC',
             'MPO'
             )
MINS = {'IPC':0.0,'MPO':0.0,}
MAXS = {'IPC':2.4,'MPO':50.0,}
BUCKETS = {'IPC':24,'MPO':25.0}

# cpu power cap
CPU = [95,100,105,110,115,120,125]
cpu_to_bucket = {CPU[i]:i for i in range(len(CPU))}
cpu_limit = 125
cpu_num = 7
CPU_ACTIONS = cpu_num
cpu_num_buckets = np.array([BUCKETS[k] for k in CPU_LABELS], dtype=np.double)
if FREQ_IN_STATE:
    cpu_dims = [int(b) for b in cpu_num_buckets] + [cpu_num ] + [CPU_ACTIONS]
else:
    cpu_dims = [int(b) for b in cpu_num_buckets] + [CPU_ACTIONS]
print(cpu_dims)
CPU_Q_A = np.zeros(cpu_dims)
CPU_Q_B = np.zeros(cpu_dims)


# uncore frequency
UF = [3090,3091,3092,3093,3094,3095,3096] # 1.8 2.4 GHz 257 514
uf_to_bucket = {UF[i]:i for i in range(len(UF))}
uncore_freq = 3090
uf_num = 7
UF_ACTIONS = uf_num
uf_num_buckets = np.array([BUCKETS[k] for k in UF_LABELS], dtype=np.double)
if FREQ_IN_STATE:
    uf_dims = [int(b) for b in uf_num_buckets] + [uf_num ] + [UF_ACTIONS]
else:
    uf_dims = [int(b) for b in uf_num_buckets] + [UF_ACTIONS]
print(uf_dims)
UF_Q_A = np.zeros(uf_dims)
UF_Q_B = np.zeros(uf_dims)

#
if os.path.isfile('Q_A_cpu_en.npy'):
    CPU_Q_A = np.load("Q_A_cpu_en.npy")
    CPU_Q_B = np.load("Q_B_cpu_en.npy")
    # UF_Q_A = np.load("Q_A_uf_en.npy")
    # UF_Q_B = np.load("Q_B_uf_en.npy")
    print("load Q table!")


# get states
# while True:
for n in range(10000):
    os.system('perf stat -e power/energy-pkg/,power/energy-ram/,instructions,LLC-load-misses,LLC-store-misses,cpu-cycles -a sleep 3 2> states.txt > states.txt')
    with open('states.txt', 'r') as fo:
        states_lines = fo.readlines()
        for states in states_lines:
            states = states.replace(',', '')
            if states.find("instructions",0,37) > 0:
                ins = float(states.split()[0])
            elif states.find("LLC-load-misses ") > 0:
                llclm = float(states.split()[0])
            elif states.find("LLC-store-misses ") > 0:
                llcsm = float(states.split()[0])
            elif states.find("cpu-cycles ") > 0:
                cyc= float(states.split()[0])
            elif states.find("power/energy-pkg/ ") > 0:
                en_pkg= float(states.split()[0])
            elif states.find("power/energy-ram/ ") > 0:
                en_ram = float(states.split()[0])
    ipc = ins / cyc
    mpo = (llclm+llcsm)/ins*10000
    ed2p = ((ipc*10)** 3) / (en_pkg/3)

    stats ={'IPC':ipc,'MPO':mpo,'E2DP':ed2p,
            # 'CPUL' :cpu_limit ,
            # 'UNCOREF': uncore_freq,
            'EN_PKG':en_pkg,
            'EN_RAM': en_ram
            }
    print(stats)

# Reward function

    reward=stats['E2DP']
    reward=round(reward)
    cpu_reward = reward
    # uf_reward = reward
    # print(uf_reward)


# CPU states
    cpu_all_mins = np.array([MINS[k] for k in CPU_LABELS], dtype=np.double)
    cpu_all_maxs = np.array([MAXS[k] for k in CPU_LABELS], dtype=np.double)
    cpu_num_buckets = np.array([BUCKETS[k] for k in CPU_LABELS], dtype=np.double)
    cpu_widths = np.divide(np.array(cpu_all_maxs) - np.array(cpu_all_mins), cpu_num_buckets)  # divide /

    cpu_raw_no_pow = [stats[k] for k in CPU_LABELS]  # wym modify
    cpu_raw_no_pow = np.clip(cpu_raw_no_pow, cpu_all_mins, cpu_all_maxs) # clip set data at range(min max)

    cpu_raw_floored = cpu_raw_no_pow - cpu_all_mins
    cpu_state = np.divide(cpu_raw_floored, cpu_widths)
    cpu_state = np.clip(cpu_state, 0, cpu_num_buckets - 1)
    if FREQ_IN_STATE:
        # Add power index to end of state:
        cpu_state = np.append(cpu_state, [cpu_to_bucket[stats['CPUL']]])
    # Convert floats to integer bucket indices and return:
    cpu_state = [int(x) for x in cpu_state]
    print("ipc:----------------------------------")
    print(cpu_state)

# # UNCORE states
#
#     uf_all_mins = np.array([MINS[k] for k in UF_LABELS], dtype=np.double)
#     uf_all_maxs = np.array([MAXS[k] for k in UF_LABELS], dtype=np.double)
#     uf_num_buckets = np.array([BUCKETS[k] for k in UF_LABELS], dtype=np.double)
#     uf_widths = np.divide(np.array(uf_all_maxs) - np.array(uf_all_mins), uf_num_buckets)  # divide /
#
#     uf_raw_no_pow = [stats[k] for k in UF_LABELS]  # wym modify
#     uf_raw_no_pow = np.clip(uf_raw_no_pow, uf_all_mins, uf_all_maxs)  # clip set data at range(min max)
#
#     uf_raw_floored = uf_raw_no_pow - uf_all_mins
#     uf_state = np.divide(uf_raw_floored, uf_widths)
#     uf_state = np.clip(uf_state, 0, uf_num_buckets - 1)
#     if FREQ_IN_STATE:
#         # Add power index to end of state:
#         uf_state = np.append(uf_state, [uf_to_bucket[stats['UNCOREF']]])
#     # Convert floats to integer bucket indices and return:
#     uf_state = [int(x) for x in uf_state]
#     print("opm---------------------------------")
#     print(uf_state)

#CPU Double Q-learning
    if cpu_last_action is not None:
        if random.random() > EPSILON:
            # best_action = np.argmax(Q_B[tuple(state)])

            if (CPU_Q_A[tuple(cpu_last_state + [cpu_last_action])] > CPU_Q_B[tuple(cpu_last_state + [cpu_last_action])]):
                cpu_best_action = np.argmax(CPU_Q_A[tuple(cpu_state)])
                print("!!!!!!!!!!! CPU Q_A !!!!!!!!!!!!!!")
            else:
                cpu_best_action = np.argmax(CPU_Q_B[tuple(cpu_state)])
                print("!!!!!!!!!!! CPU Q_B !!!!!!!!!!!!!!")

            # if (UF_Q_A[tuple(uf_last_state + [uf_last_action])] > UF_Q_B[tuple(uf_last_state + [uf_last_action])]):
            #     uf_best_action = np.argmax(UF_Q_A[tuple(uf_state)])
            #     print("!!!!!!!!!!! Q_A !!!!!!!!!!!!!!")
            # else:
            #     uf_best_action = np.argmax(UF_Q_B[tuple(uf_state)])
            #     print("!!!!!!!!!!! Q_B !!!!!!!!!!!!!!")
        else:
            cpu_best_action = random.randint(0, CPU_ACTIONS - 1)
            print("!!!!!!!!!!! 2 select random !!!!!!!!!!!!!!")

            # uf_best_action = random.randint(0, UF_ACTIONS - 1)
            # # print("!!!!!!!!!!! 2 select random !!!!!!!!!!!!!!")
    else:
        cpu_best_action = random.randint(0, CPU_ACTIONS - 1)
        # print("!!!!!!!!!!! 1 select random !!!!!!!!!!!!!!")
        #
        # uf_best_action = random.randint(0, UF_ACTIONS - 1)
        # # print("!!!!!!!!!!! 1 select random !!!!!!!!!!!!!!")

    cpu_last_state = cpu_state
    cpu_last_action = cpu_best_action
    # print(cpu_last_state)
    # print(cpu_last_action)
    # take action
    cpu_limit= CPU[cpu_last_action]
    print(cpu_limit)
    os.system("./set_pkg_limit %s " % (cpu_limit))

    # uf_last_state = uf_state
    # uf_last_action = uf_best_action
    # # print(uf_last_state)
    # # print(uf_last_action)
    # # take action
    # uncore_freq = UF[uf_last_action]
    # print(uncore_freq)
    # os.system("./set_uncore_limit %s " % (uncore_freq))


    if random.random() > EPSILON1:
        cpu_best_action_A = np.argmax(CPU_Q_A[tuple(cpu_state)])
        cpu_old_value_B = CPU_Q_B[tuple(cpu_state + [cpu_best_action_A])]
        cpu_total_return_B = cpu_reward + GAMMA * cpu_old_value_B
        cpu_old_value_A = CPU_Q_A[tuple(cpu_last_state + [cpu_last_action])]
        CPU_Q_A[tuple(cpu_last_state + [cpu_last_action])] = cpu_old_value_A + ALPHA * (
                    cpu_total_return_B - cpu_old_value_A)
    # best_action = np.argmax(Q_A[tuple(state)])
    #     uf_best_action_A = np.argmax(UF_Q_A[tuple(uf_state)])
    #     uf_old_value_B = UF_Q_B[tuple(uf_state + [uf_best_action_A])]
    #     uf_total_return_B = uf_reward + GAMMA * uf_old_value_B
    #     uf_old_value_A = UF_Q_A[tuple(uf_last_state + [uf_last_action])]
    #     UF_Q_A[tuple(uf_last_state + [uf_last_action])] = uf_old_value_A + ALPHA * (uf_total_return_B - uf_old_value_A)
    else:
        cpu_best_action_B = np.argmax(CPU_Q_B[tuple(cpu_state)])
        cpu_old_value_A = CPU_Q_A[tuple(cpu_state + [cpu_best_action_B])]
        cpu_total_return_A = cpu_reward + GAMMA * cpu_old_value_A
        cpu_old_value_B = CPU_Q_B[tuple(cpu_last_state + [cpu_last_action])]
        CPU_Q_B[tuple(cpu_last_state + [cpu_last_action])] = cpu_old_value_B + ALPHA * (
                    cpu_total_return_A - cpu_old_value_B)

        # uf_best_action_B = np.argmax(UF_Q_B[tuple(uf_state)])
        # uf_old_value_A = UF_Q_A[tuple(uf_state + [uf_best_action_B])]
        # uf_total_return_A = uf_reward + GAMMA * uf_old_value_A
        # uf_old_value_B = UF_Q_B[tuple(uf_last_state + [uf_last_action])]
        # UF_Q_B[tuple(uf_last_state + [uf_last_action])] = uf_old_value_B + ALPHA * (uf_total_return_A - uf_old_value_B)


    print("--------------------------------------",n,"-----------------------------------")
print("-----------------------------------   end    -----------------------------------------------------")

np.save(file="Q_A_cpu_en.npy", arr=CPU_Q_A)
np.save(file="Q_B_cpu_en.npy", arr=CPU_Q_B)
# np.save(file="Q_A_uf_en.npy", arr=UF_Q_A)
# np.save(file="Q_B_uf_en.npy", arr=UF_Q_B)
