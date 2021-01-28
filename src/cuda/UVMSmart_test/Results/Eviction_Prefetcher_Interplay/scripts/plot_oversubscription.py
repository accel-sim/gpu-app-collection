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

experiment_folder = 'Oversubscription'
sub_folders = ['NoOversub', '110', '125', '110_PreEvict95', '110_PreEvict90']

benchmarks = ['backprop', 'bfs', 'fdtd', 'hotspot', 'nw', 'pathfinder', 'srad']

rt_NoOversub = []
rt_110 = []
rt_125 = []
rt_110_PreEvict95 = []
rt_110_PreEvict90 = []

pt4k_NoOversub = []
pt4k_110 = []
pt4k_125 = []
pt4k_110_PreEvict95 = []
pt4k_110_PreEvict90 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'NoOversub':
				rt_NoOversub.append(rt)
				pt4k_NoOversub.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '110':
				rt_110.append(rt)
				pt4k_110.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '125':
				rt_125.append(rt)
				pt4k_125.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '110_PreEvict95':
				rt_110_PreEvict95.append(rt)
				pt4k_110_PreEvict95.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '110_PreEvict90':
				rt_110_PreEvict90.append(rt)
				pt4k_110_PreEvict90.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			

rt_110 = np.array(np.divide(rt_110, rt_NoOversub))
rt_125 = np.array(np.divide(rt_125, rt_NoOversub))
rt_110_PreEvict95 = np.array(np.divide(rt_110_PreEvict95, rt_NoOversub))
rt_110_PreEvict90 = np.array(np.divide(rt_110_PreEvict90, rt_NoOversub))

pt4k_110 = np.array(np.divide(pt4k_110, pt4k_NoOversub))
pt4k_125 = np.array(np.divide(pt4k_125, pt4k_NoOversub))
pt4k_110_PreEvict95 = np.array(np.divide(pt4k_110_PreEvict95, pt4k_NoOversub))
pt4k_110_PreEvict90 = np.array(np.divide(pt4k_110_PreEvict90, pt4k_NoOversub))

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.3
 
# Set position of bar on X axis
r1 = np.arange(len(rt_NoOversub), dtype=float)

r1a = [0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6]

for i in range(len(rt_NoOversub)):
	r1[i] = r1[i] + 0.3

for i in range(len(rt_NoOversub)):
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

plt.bar(r2, rt_110, hatch="--", color='r', width=barWidth, edgecolor='black', label='Working set == device memory size * 110%')
plt.bar(r3, rt_125, hatch="++", color='c', width=barWidth, edgecolor='black', label='Working set == device memory size * 125%')         
plt.bar(r4, rt_110_PreEvict95, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='Working set == device memory size * 110% and pre-eviction at 95%')
plt.bar(r5, rt_110_PreEvict90, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='Working set == device memory size * 110% and pre-eviction at 90%')

plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Kernel Execution Time\n(Normalized to no oversubscription)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), prop={'size': 12})

plt.savefig('../plots/Oversubscription/oversub.png',  dpi=300, bbox_inches="tight")

# Child plot
plt.cla()
plt.clf()

plt.rc('font', **font)

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(10,5))

plt.bar(r2, pt4k_110, hatch="--", color='r', width=barWidth, edgecolor='black', label='Working set == device memory size * 110%')
plt.bar(r3, pt4k_125, hatch="++", color='c', width=barWidth, edgecolor='black', label='Working set == device memory size * 125%')         
plt.bar(r4, pt4k_110_PreEvict95, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='Working set == device memory size * 110% and pre-eviction at 95%')
plt.bar(r5, pt4k_110_PreEvict90, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='Working set == device memory size * 110% and pre-eviction at 90%')

plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Number of 4KB Page Transfers\n(Normalized to no oversubscription)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), prop={'size': 12})

plt.savefig('../plots/Oversubscription/oversub_4k.png',  dpi=300, bbox_inches="tight")

