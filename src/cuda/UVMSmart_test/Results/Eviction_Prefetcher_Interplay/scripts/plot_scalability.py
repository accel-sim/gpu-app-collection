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

experiment_folder = 'Scalability'
sub_folders = ['110', '125', '150']

benchmarks = ['backprop', 'bfs', 'fdtd', 'hotspot', 'nw', 'pathfinder', 'srad']

rt_110 = []
rt_125 = []
rt_150 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == '110':
				rt_110.append(rt)
			elif sf == '125':
				rt_125.append(rt)
			elif sf == '150':
				rt_150.append(rt)			

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.3
 
# Set position of bar on X axis
r1 = np.arange(len(rt_110), dtype=float)

r1a = [0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6]

for i in range(len(rt_110)):
	r1[i] = r1[i] + 0.3

for i in range(len(rt_110)):
	r1a[i] = r1a[i] + 0.3

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

plt.bar(r2, rt_110, hatch="--", color='r', width=barWidth, edgecolor='black', label='Working set == device memory size * 110%')
plt.bar(r3, rt_125, hatch="++", color='c', width=barWidth, edgecolor='black', label='Working set == device memory size * 125%')         
plt.bar(r4, rt_150, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='Working set == device memory size * 150%')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

#ax.axes.set_ylim([0,100000])
#ax.set_yscale('log', nonposy='clip')

plt.ylabel('Kernel Execution Time (us)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), prop={'size': 12})

plt.savefig('../plots/Scalability/scalability.png',  dpi=300, bbox_inches="tight")
