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

experiment_folder = 'HWPrefetchers'
sub_folders = ['TBN', 'SL', 'Random', 'OnDemand']

benchmarks = ['backprop', 'bfs', 'fdtd', 'hotspot', 'nw', 'pathfinder', 'srad']

rt_OnDemand = []
rt_TBN = []
rt_SL = []
rt_Random = []

rbw_OnDemand = []
rbw_TBN = []
rbw_SL = []
rbw_Random = []

pf_OnDemand = []
pf_TBN = []
pf_SL = []
pf_Random = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			line = re.findall(r"Pcie_read_utilization.*", file_content, flags=re.MULTILINE)[0]
			rbw = float(line[line.find(': ')+2:len(line)]) * 12

			line = re.findall(r"Total_memory_access_page_fault.*", file_content, flags=re.MULTILINE)[0]
			pf = float(line[line.find(': ')+2:line.find(', ')])

			if sf == 'TBN':
				rt_TBN.append(rt)
				rbw_TBN.append(rbw)
				pf_TBN.append(pf)
			elif sf == 'SL':
				rt_SL.append(rt)
				rbw_SL.append(rbw)
				pf_SL.append(pf)
			elif sf == 'Random':
				rt_Random.append(rt)
				rbw_Random.append(rbw)
				pf_Random.append(pf)
			elif sf == 'OnDemand':
				rt_OnDemand.append(rt)
				rbw_OnDemand.append(rbw)
				pf_OnDemand.append(pf)			


#######################
# plotting section
#######################

# set width of bar
barWidth = 0.3
 
# Set position of bar on X axis
r1 = np.arange(len(rt_TBN), dtype=float)

r1a = [0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6]

for i in range(len(rt_TBN)):
	r1[i] = r1[i] + 0.3

for i in range(len(rt_TBN)):
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

plt.bar(r2, rt_OnDemand, hatch="--", color='r', width=barWidth, edgecolor='black', label='No prefetcher')
plt.bar(r3, rt_Random, hatch="++", color='c', width=barWidth, edgecolor='black', label='$R_p$')         
plt.bar(r4, rt_SL, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='$SL_p$')
plt.bar(r5, rt_TBN, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='$TBN_p$')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax.set_yscale('log', nonposy='clip')

plt.ylabel('Kernel Execution Time (log scale)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), prop={'size': 12}, ncol=4)

plt.savefig('../plots/HWPrefetchers/hwprefetchers.png',  dpi=300, bbox_inches="tight")

# Child plot

plt.cla()
plt.clf()

plt.figure(2)

plt.rc('font', **font)

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(10,5))

plt.bar(r2, rbw_OnDemand, hatch="--", color='r', width=barWidth, edgecolor='black', label='No prefetcher')
plt.bar(r3, rbw_Random, hatch="++", color='c', width=barWidth, edgecolor='black', label='$R_p$')         
plt.bar(r4, rbw_SL, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='$SL_p$')
plt.bar(r5, rbw_TBN, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='$TBN_p$')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Average PCI-e Read Bandwidth (GB/s)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), prop={'size': 12}, ncol=4)

plt.savefig('../plots/HWPrefetchers/hwprefetchers_read_bw.png',  dpi=300, bbox_inches="tight")

# Child plot 2

plt.cla()
plt.clf()

plt.figure(3)

plt.rc('font', **font)

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(10,5))

plt.bar(r2, pf_OnDemand, hatch="--", color='r', width=barWidth, edgecolor='black', label='No prefetcher')
plt.bar(r3, pf_Random, hatch="++", color='c', width=barWidth, edgecolor='black', label='$R_p$')
plt.bar(r4, pf_SL, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='$SL_p$')
plt.bar(r5, pf_TBN, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='$TBN_p$')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax.set_yscale('log', nonposy='clip')

plt.ylabel('Total Number of Far-faults (log scale)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), prop={'size': 12}, ncol=4)

plt.savefig('../plots/HWPrefetchers/hwprefetchers_page_fault.png',  dpi=300, bbox_inches="tight")

