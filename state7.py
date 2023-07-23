'''
2020.8.11
pkg power capping +DRAM power capping + uncore frequency
CPU_Q
DRAM_Q
UF_Q
'''
import numpy as np
import random
import os
'''
reward:		power/energy-pkg/ 
State space:
0	insn per cycle	=  instructions/ cpu-cycles
1	L1-dcache-load-misses / L1-dcache-loads
2	LLC-load-misses / LLC-loads
3	LLC-store-misses / LLC-stores 
4	branch-misses / branch-instructions
5	cache-misses /  cache-references 
6   MPO
'''
FREQ_IN_STATE=1
cpu_last_action = None
cpu_last_state = None
dram_last_action = None
dram_last_state = None
uf_last_action = None
uf_last_state = None
EPSILON = 0.15         # greedy police  train：0.15  test：0.15 test1:0.01
GAMMA = 0.95            # Discounting factor  0.95
ALPHA = 0.7           # learning rate  0.5~0.9 0.7-> bad   0.1
'''
q-learning  init  runtime vars
'''
CPU_LABELS = ('IPC',
          # 'L1',
          #'LCL',
          #'LCS'
          # 'LC'
          #'MPO',
         # 'BRA'
         #'CACE'
          )
# DRAM_LABELS = (#'IPC',
#           # 'L1',
#           # 'LCL',
#           # 'LCS'
#           'LC',
#           'MPO'
#          # 'BRA'
#          #'CACE'
#           )
UF_LABELS = (#'IPC',
          # 'L1',
          # 'LCL',
          # 'LCS',
          #'LC'
          'MPO',
         # 'BRA'
         #'CACE'
          )
MINS = {'IPC':0.0,'L1':0.0,'LCL':0.0, 'LCS':0.0,'LC':0.0,'MPO':0.0,'BRA':0.0,'CACE':0.0}
MAXS = {'IPC':2.0,'L1':1.0,'LCL':1.0, 'LCS':1.0,'LC':1.0,'MPO':30.0,'BRA':1.0,'CACE':1.0}
BUCKETS = {'IPC':20, 'L1':10,'LCL':10, 'LCS':10,'LC':10.0,'MPO':30.0,'BRA':10.0,'CACE':10.0}

cpu#  power cap
CPU = [65,70,75,80,85,90,95,100,105,110,115,120,125]
cpu_to_bucket = {CPU[i]:i for i in range(len(CPU))}
cpu_limit = 75
cpu_num = 13
CPU_ACTIONS = cpu_num
cpu_num_buckets = np.array([BUCKETS[k] for k in CPU_LABELS], dtype=np.double)
if FREQ_IN_STATE:
    cpu_dims = [int(b) for b in cpu_num_buckets] + [cpu_num ] + [CPU_ACTIONS]
else:
    cpu_dims = [int(b) for b in cpu_num_buckets] + [CPU_ACTIONS]
print(cpu_dims)
CPU_Q_A = np.zeros(cpu_dims)
CPU_Q_B = np.zeros(cpu_dims)

# dram power cap
# DRAM = [20,26,32]
# dram_to_bucket = {DRAM[i]:i for i in range(len(DRAM))}
# dram_limit = 20
# dram_num = 3
# DRAM_ACTIONS = dram_num
# dram_num_buckets = np.array([BUCKETS[k] for k in DRAM_LABELS], dtype=np.double)
# if FREQ_IN_STATE:
#     dram_dims = [int(b) for b in dram_num_buckets] + [dram_num ] + [DRAM_ACTIONS]
# else:
#     dram_dims = [int(b) for b in dram_num_buckets] + [DRAM_ACTIONS]
# DRAM_Q_A = np.zeros(dram_dims)
# DRAM_Q_B = np.zeros(dram_dims)

# uncore frequency
UF = list(range(3084,6425,257)) # 1.2 2.4 GHz 257 514
uf_to_bucket = {UF[i]:i for i in range(len(UF))}
uncore_freq = 3084
uf_num = 13
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
    UF_Q_A = np.load("Q_A_uf_en.npy")
    UF_Q_B = np.load("Q_B_uf_en.npy")
    print("load Q table!")
# Note: C is no longer used to keep track of exact counts, but if a state action has been seen ever.

# get states
# while True:
for n in range(6000):
    os.system('perf stat -e power/energy-pkg/,power/energy-ram/,instructions,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,cpu-cycles -a sleep 3 2> states.txt > states.txt')
    with open('states.txt', 'r') as fo:
        states_lines = fo.readlines()
        for states in states_lines:
            states = states.replace(',', '')
            if states.find("instructions",0,37) > 0:
                ins = float(states.split()[0])
            elif states.find("LLC-load-misses ") > 0:
                llclm = float(states.split()[0])
            elif states.find("LLC-loads ") > 0:
                llcl = float(states.split()[0])
            elif states.find("LLC-store-misses ") > 0:
                llcsm = float(states.split()[0])
            elif states.find("LLC-stores  ") > 0:
                llcs = float(states.split()[0])
            elif states.find("cpu-cycles ") > 0:
                cyc= float(states.split()[0])
            elif states.find("power/energy-pkg/ ") > 0:
                en_pkg= float(states.split()[0])
            elif states.find("power/energy-ram/ ") > 0:
                en_ram = float(states.split()[0])
    ipc = ins / cyc
    lcl = llclm / llcl
    lcs = llcsm / llcs
    lc = (llclm+llcsm) /(llcl+llcs)
    mpo = (llclm+llcsm)/ins*10000
    cpul = cpu_limit
    # draml = dram_limit
    uncoref = uncore_freq
    stats ={'IPC':ipc,'LCL':lcl,'LCS':lcs,'MPO':mpo,'LC':lc,
            'CPUL' :cpul ,
            #'DRAML': draml,
            'UNCOREF': uncoref,
            'EN_PKG':en_pkg,'EN_RAM':en_ram }
    print(stats)

# Reward function
    ener_pkg = stats['EN_PKG']
    ener_ram = stats['EN_RAM']
    # ener_pkg = round(en_pkg)
    # ener_ram = round(en_ram)
    ener = ener_pkg + ener_ram

    # cpu_reward = -(ener_pkg / 10)
    # # dram_reward = -ener_ram
    # uf_reward = -(ener_pkg / 10)  # en
    reward = -(ener / 10)
    # cpu_reward = round(cpu_reward)
    # uf_reward = round(uf_reward)
    cpu_reward = round(reward)
    uf_reward = round(reward)

    # cpu_reward = ener_pkg * 0.1   #EDP
    # uf_reward = ener_pkg * 0.1

    print("cpu : ")
    print(cpu_reward)
    print("uf : ")
    print(uf_reward)

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

# DRAM states
#     dram_all_mins = np.array([MINS[k] for k in DRAM_LABELS], dtype=np.double)
#     dram_all_maxs = np.array([MAXS[k] for k in DRAM_LABELS], dtype=np.double)
#     dram_num_buckets = np.array([BUCKETS[k] for k in DRAM_LABELS], dtype=np.double)
#     dram_widths = np.divide(np.array(dram_all_maxs) - np.array(dram_all_mins), dram_num_buckets)  # divide /
#
#     dram_raw_no_pow = [stats[k] for k in DRAM_LABELS]  # wym modify
#     dram_raw_no_pow = np.clip(dram_raw_no_pow, dram_all_mins, dram_all_maxs)  # clip set data at range(min max)
#
#     dram_raw_floored = dram_raw_no_pow - dram_all_mins
#     dram_state = np.divide(dram_raw_floored, dram_widths)
#     dram_state = np.clip(dram_state, 0, dram_num_buckets - 1)
#     if FREQ_IN_STATE:
#         # Add power index to end of state:
#         dram_state = np.append(dram_state, [dram_to_bucket[stats['DRAML']]])
#     # Convert floats to integer bucket indices and return:
#     dram_state = [int(x) for x in dram_state]

# UNCORE states
    uf_all_mins = np.array([MINS[k] for k in UF_LABELS], dtype=np.double)
    uf_all_maxs = np.array([MAXS[k] for k in UF_LABELS], dtype=np.double)
    uf_num_buckets = np.array([BUCKETS[k] for k in UF_LABELS], dtype=np.double)
    uf_widths = np.divide(np.array(uf_all_maxs) - np.array(uf_all_mins), uf_num_buckets)  # divide /

    uf_raw_no_pow = [stats[k] for k in UF_LABELS]  # wym modify
    uf_raw_no_pow = np.clip(uf_raw_no_pow, uf_all_mins, uf_all_maxs)  # clip set data at range(min max)

    uf_raw_floored = uf_raw_no_pow - uf_all_mins
    uf_state = np.divide(uf_raw_floored, uf_widths)
    uf_state = np.clip(uf_state, 0, uf_num_buckets - 1)
    if FREQ_IN_STATE:
        # Add power index to end of state:
        uf_state = np.append(uf_state, [uf_to_bucket[stats['UNCOREF']]])
    # Convert floats to integer bucket indices and return:
    uf_state = [int(x) for x in uf_state]

#CPU Double Q-learning
    if cpu_last_action is not None:
        if random.random() > EPSILON:
            cpu_best_action_A = np.argmax(CPU_Q_A[tuple(cpu_state)])
            cpu_old_value_B = CPU_Q_B[tuple(cpu_state + [cpu_best_action_A])]
            cpu_total_return_B = cpu_reward + GAMMA *  cpu_old_value_B
            cpu_old_value_A = CPU_Q_A[tuple(cpu_last_state + [cpu_last_action])]
            CPU_Q_A[tuple(cpu_last_state + [cpu_last_action])] = cpu_old_value_A + ALPHA * (cpu_total_return_B - cpu_old_value_A)
            # best_action = np.argmax(Q_A[tuple(state)])

            cpu_best_action_B = np.argmax(CPU_Q_B[tuple(cpu_state)])
            cpu_old_value_A = CPU_Q_A[tuple(cpu_state + [cpu_best_action_B])]
            cpu_total_return_A = cpu_reward + GAMMA * cpu_old_value_A
            cpu_old_value_B = CPU_Q_B[tuple(cpu_last_state + [cpu_last_action])]
            CPU_Q_B[tuple(cpu_last_state + [cpu_last_action])] = cpu_old_value_B + ALPHA * (cpu_total_return_A - cpu_old_value_B)
            # best_action = np.argmax(Q_B[tuple(state)])

            if (CPU_Q_A[tuple(cpu_last_state + [cpu_last_action])] > CPU_Q_B[tuple(cpu_last_state + [cpu_last_action])]):
                cpu_best_action = np.argmax(CPU_Q_A[tuple(cpu_state)])
                print("!!!!!!!!!!! Q_A !!!!!!!!!!!!!!")
            else:
                cpu_best_action = np.argmax(CPU_Q_B[tuple(cpu_state)])
                print("!!!!!!!!!!! Q_B !!!!!!!!!!!!!!")
        else:
            cpu_best_action = random.randint(0, CPU_ACTIONS - 1)
            print("!!!!!!!!!!! 2 select random !!!!!!!!!!!!!!")
    else:
        cpu_best_action = random.randint(0, CPU_ACTIONS - 1)
        print("!!!!!!!!!!! 1 select random !!!!!!!!!!!!!!")
    cpu_last_state = cpu_state
    cpu_last_action = cpu_best_action
    print(cpu_last_state)
    print(cpu_last_action)
    # take action
    cpul= CPU[cpu_last_action]
    print(cpul)
    os.system("./set_pkg_limit %s " % (cpul))

# DRAM Double Q-learning
#     if dram_last_action is not None:
#         if random.random() > EPSILON:
#             dram_best_action_A = np.argmax(DRAM_Q_A[tuple(dram_state)])
#             dram_old_value_B = DRAM_Q_B[tuple(dram_state + [dram_best_action_A])]
#             dram_total_return_B = dram_reward + GAMMA * dram_old_value_B
#             dram_old_value_A = DRAM_Q_A[tuple(dram_last_state + [dram_last_action])]
#             DRAM_Q_A[tuple(dram_last_state + [dram_last_action])] = dram_old_value_A + ALPHA * (
#                         dram_total_return_B - dram_old_value_A)
#             # best_action = np.argmax(Q_A[tuple(state)])
#
#             dram_best_action_B = np.argmax(DRAM_Q_B[tuple(dram_state)])
#             dram_old_value_A = DRAM_Q_A[tuple(dram_state + [dram_best_action_B])]
#             dram_total_return_A = dram_reward + GAMMA * dram_old_value_A
#             dram_old_value_B = DRAM_Q_B[tuple(dram_last_state + [dram_last_action])]
#             DRAM_Q_B[tuple(dram_last_state + [dram_last_action])] = dram_old_value_B + ALPHA * (
#                         dram_total_return_A - dram_old_value_B)
#             # best_action = np.argmax(Q_B[tuple(state)])
#
#             if (DRAM_Q_A[tuple(dram_last_state + [dram_last_action])] > DRAM_Q_B[tuple(dram_last_state + [dram_last_action])]):
#                 dram_best_action = np.argmax(DRAM_Q_A[tuple(dram_state)])
#                 print("!!!!!!!!!!! Q_A !!!!!!!!!!!!!!")
#             else:
#                 dram_best_action = np.argmax(DRAM_Q_B[tuple(dram_state)])
#                 print("!!!!!!!!!!! Q_B !!!!!!!!!!!!!!")
#         else:
#             dram_best_action = random.randint(0, DRAM_ACTIONS - 1)
#             print("!!!!!!!!!!! 2 select random !!!!!!!!!!!!!!")
#     else:
#         dram_best_action = random.randint(0, DRAM_ACTIONS - 1)
#         print("!!!!!!!!!!! 1 select random !!!!!!!!!!!!!!")
#     dram_last_state = dram_state
#     dram_last_action = dram_best_action
#     print(dram_last_state)
#     print(dram_last_action)
#     # take action
#     draml = DRAM[dram_last_action]
#     print(draml)
#     os.system("./set_dram_limit %s " % (draml))

# UNCORE Double Q-learning
    if uf_last_action is not None:
        if random.random() > EPSILON:
            uf_best_action_A = np.argmax(UF_Q_A[tuple(uf_state)])
            uf_old_value_B = UF_Q_B[tuple(uf_state + [uf_best_action_A])]
            uf_total_return_B = uf_reward + GAMMA *  uf_old_value_B
            uf_old_value_A = UF_Q_A[tuple(uf_last_state + [uf_last_action])]
            UF_Q_A[tuple(uf_last_state + [uf_last_action])] = uf_old_value_A + ALPHA * (uf_total_return_B - uf_old_value_A)
            # best_action = np.argmax(Q_A[tuple(state)])

            uf_best_action_B = np.argmax(UF_Q_B[tuple(uf_state)])
            uf_old_value_A = UF_Q_A[tuple(uf_state + [uf_best_action_B])]
            uf_total_return_A = uf_reward + GAMMA * uf_old_value_A
            uf_old_value_B = UF_Q_B[tuple(uf_last_state + [uf_last_action])]
            UF_Q_B[tuple(uf_last_state + [uf_last_action])] = uf_old_value_B + ALPHA * (uf_total_return_A - uf_old_value_B)
            # best_action = np.argmax(Q_B[tuple(state)])

            if (UF_Q_A[tuple(uf_last_state + [uf_last_action])] > UF_Q_B[tuple(uf_last_state + [uf_last_action])]):
                uf_best_action = np.argmax(UF_Q_A[tuple(uf_state)])
                print("!!!!!!!!!!! Q_A !!!!!!!!!!!!!!")
            else:
                uf_best_action = np.argmax(UF_Q_B[tuple(uf_state)])
                print("!!!!!!!!!!! Q_B !!!!!!!!!!!!!!")
        else:
            uf_best_action = random.randint(0, UF_ACTIONS - 1)
            print("!!!!!!!!!!! 2 select random !!!!!!!!!!!!!!")
    else:
        uf_best_action = random.randint(0, UF_ACTIONS - 1)
        print("!!!!!!!!!!! 1 select random !!!!!!!!!!!!!!")
    uf_last_state = uf_state
    uf_last_action = uf_best_action
    print(uf_last_state)
    print(uf_last_action)
    # take action
    uncoref = UF[uf_last_action]
    print(uncoref)
    os.system("./set_uncore_limit %s " % (uncoref))


    print("--------------------------------------",n,"-----------------------------------")
print("-----------------------------------   end    -----------------------------------------------------")

np.save(file="Q_A_cpu_en.npy", arr=CPU_Q_A)
np.save(file="Q_B_cpu_en.npy", arr=CPU_Q_B)
np.save(file="Q_A_uf_en.npy", arr=UF_Q_A)
np.save(file="Q_B_uf_en.npy", arr=UF_Q_B)