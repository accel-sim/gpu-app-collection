# libraries
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'

experiment_folder = 'EvictionPolicy_HWPrefetcher_Combo'
sub_folders = ['LRU_OD', 'Random_Random', 'SL_SL', 'TBN_TBN']

benchmarks = ['backprop', 'bfs', 'fdtd', 'hotspot', 'nw', 'pathfinder', 'srad']

rt_LRU_OD = []
rt_Random_Random = []
rt_SL_SL = []
rt_TBN_TBN = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'LRU_OD':
				rt_LRU_OD.append(rt)
			elif sf == 'Random_Random':
				rt_Random_Random.append(rt)
			elif sf == 'SL_SL':
				rt_SL_SL.append(rt)
			elif sf == 'TBN_TBN':
				rt_TBN_TBN.append(rt)			

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.3
 
# Set position of bar on X axis
r1 = np.arange(len(rt_LRU_OD), dtype=float)

r1a = [0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6]

for i in range(len(rt_LRU_OD)):
	r1[i] = r1[i] + 0.3

for i in range(len(rt_LRU_OD)):
	r1a[i] = r1a[i] + 0.3

r2 = [x + barWidth for x in r1a]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]


plt.figure(1)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(10,5))

plt.bar(r2, rt_LRU_OD, hatch="--", color='r', width=barWidth, edgecolor='black', label='LRU 4KB + No prefetcher')
plt.bar(r3, rt_Random_Random, hatch="++", color='c', width=barWidth, edgecolor='black', label='$R_e$ + $R_p$')         
plt.bar(r4, rt_SL_SL, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='$SL_e$ + $SL_p$')
plt.bar(r5, rt_TBN_TBN, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='$TBN_e$ + $TBN_p$')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax.set_yscale('log', nonposy='clip')

plt.ylabel('Kernel Execution Time (log scale)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), prop={'size': 12}, ncol=4)

plt.savefig('../plots/EvictionPolicy_HWPrefetcher_Combo/combo.png',  dpi=300, bbox_inches="tight")
