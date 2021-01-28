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

experiment_folder = 'ReserveLRU'
sub_folders = ['TBN_110', 'TBN_110_Rsv10', 'TBN_110_Rsv20']

benchmarks = ['backprop', 'bfs', 'fdtd', 'hotspot', 'nw', 'pathfinder', 'srad']

rt_TBN_110 = []
rt_TBN_110_Rsv10 = []
rt_TBN_110_Rsv20 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'TBN_110':
				rt_TBN_110.append(rt)
			elif sf == 'TBN_110_Rsv10':
				rt_TBN_110_Rsv10.append(rt)
			elif sf == 'TBN_110_Rsv20':
				rt_TBN_110_Rsv20.append(rt)

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(rt_TBN_110), dtype=float)

r1a = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2]

for i in range(len(rt_TBN_110)):
	r1[i] = r1[i] + 0.2

for i in range(len(rt_TBN_110)):
	r1a[i] = r1a[i] + 0.2

r2 = [x + barWidth for x in r1a]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]


plt.figure(1)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(10,5))

plt.bar(r2, rt_TBN_110, hatch="--", color='r', width=barWidth, edgecolor='black', label='$TBN_e$ + $TBN_p$')
plt.bar(r3, rt_TBN_110_Rsv10, hatch="++", color='c', width=barWidth, edgecolor='black', label='$TBN_e$ + $TBN_p$ + Reserve top 10% of LRU queue')
plt.bar(r4, rt_TBN_110_Rsv20, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='$TBN_e$ + $TBN_p$ + Reserve top 20% of LRU queue')

plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Kernel Execution Time (us)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), prop={'size': 12})

plt.savefig('../plots/ReserveLRU/reservelru.png',  dpi=300, bbox_inches="tight")

