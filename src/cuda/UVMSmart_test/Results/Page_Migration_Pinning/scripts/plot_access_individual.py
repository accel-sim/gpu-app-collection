import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

# Declare directory and file structure
benchmark = sys.argv[1]

log_dir = '../OutputLogs/AccessPattern/'
plot_dir = '../Plots/AccessPattern/' + benchmark + '/'

if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)

access_file = log_dir + benchmark + '_access.txt'
alloc_file = log_dir + benchmark + '_managed.txt'
kernel_file = log_dir + benchmark + '_kernel.txt'

hist_fig = plot_dir + benchmark + '_hist.png'

acc_dir = plot_dir + 'Iteartions/'

if not os.path.exists(acc_dir):
	os.makedirs(acc_dir)

# Start of managed allocation used for filtering access
managed_start = 0xc0000000 >> 12

# Read managed allocation details: DataStructure, StartAddr, Size
alloc_data = pd.read_csv(alloc_file, delim_whitespace=True)

# Derive columns by converting start address of each allocation from hex string to decimal and shifting by 12
# and then adding total number of pages (size divided by 4KB)
alloc_data['PageStart'] = alloc_data['StartAddr'].apply(lambda x: int(x, 16) >> 12)
alloc_data['PageEnd'] = (alloc_data['PageStart'] + alloc_data['Size'] / 4096).apply(int)

# Read kernel launches information: Iteration, Kernel, Start, End
kernel_data = pd.read_csv(kernel_file, delim_whitespace=True)

# Read access pattern: PageNum, MemAddr, Size, Cycle, RD, SM Id, Warp Id
acc_data = pd.read_csv(access_file, delim_whitespace=True).query('PageNum >= @managed_start')

# Derive count of each pages and sum of RD (1 = RD, 0 = WR) which will be used to find read only, write only, and read/write pages
acc_count = acc_data[['PageNum', 'RD']].groupby(['PageNum'])['RD'].count().reset_index(name='Count').sort_values(['PageNum'])
acc_sum = acc_data[['PageNum', 'RD']].groupby(['PageNum'])['RD'].sum().reset_index(name='Sum').sort_values(['PageNum'])

# Get the histogram data from derived count and sum
hist_data = pd.merge(acc_count, acc_sum, on='PageNum')

# Derive the bounds of page numbers axis by finding smallest and largest page numbers of all managed allocations and give space of 100 on both end
page_min = alloc_data['PageStart'].min() - 100
page_max = alloc_data['PageEnd'].max() + 100

# Derive the tick positions of page numbers x-axis which will be starting and ending of managed allocations
managed_bounds = np.unique(np.append(alloc_data['PageStart'].values, alloc_data['PageEnd'].values))

# Derive tick positions of managed data structure axis which will be middle of each managed allocations
alloc_mids = ((alloc_data['PageStart'] + alloc_data['PageEnd'])/2).apply(int).get_values()

# Labels of managed data structure axis is the name of the allocations
alloc_names = alloc_data['DataStructure'].get_values()

# Labels of kernel axis is the name of the kernels
kernel_names = kernel_data['Kernel'].unique()

# Create figure for histogram 
fig = plt.figure(figsize=(10,5))

# Set font of the plot
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

# Set axes labels
plt.xlabel('Page Number')
plt.ylabel('Number of accesses')

# Create the primary x-axis
prim_ax = fig.add_subplot(111)

# Set the bounds, ticks, tick labels, and vertical grids for primary x-axis
prim_ax.set_xlim([page_min, page_max]) 
prim_ax.set_xticks(managed_bounds)
prim_ax.set_xticklabels(managed_bounds)
prim_ax.grid(b=True, which='major', color='grey', linestyle='-')

# Query comparing Sum and Count to find read only, write only, and read/write pages
# Check for empty query values
if not hist_data.query('Sum == Count').empty:
	plt.bar(hist_data.query('Sum == Count')['PageNum'].get_values(), hist_data.query('Sum == Count')['Count'].get_values(), color='r', edgecolor='none', label='Read only')

if not hist_data.query('Sum == 0').empty:
	plt.bar(hist_data.query('Sum == 0')['PageNum'].get_values(), hist_data.query('Sum == 0')['Count'].get_values(), color='k', edgecolor='none', label='Write only')

if not hist_data.query('Sum > 0 & Sum < Count').empty:
	plt.bar(hist_data.query('Sum > 0 & Sum < Count')['PageNum'].get_values(), hist_data.query('Sum > 0 & Sum < Count')['Count'].get_values(), color='b', edgecolor='none', label='Both read and write')

# Set the legends for plot
prim_ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), prop={'size': 12})

# Create the secondary x-axis
sec_ax = prim_ax.twiny()

# Set the bounds, ticks, tick labels for secondary x-axis
sec_ax.set_xlim([page_min, page_max])
sec_ax.set_xticks(alloc_mids)
sec_ax.set_xticklabels(alloc_names)

# Save the histogram 
plt.savefig(hist_fig,  dpi=300, bbox_inches="tight")

# Clear axes and figure for plot
plt.cla()
plt.clf()

# Loop on kernel iterations to plot access pattern
for iter in kernel_data['Iteration'].unique():
	cycle_min = kernel_data.query('Iteration == @iter')['Start'].min()
	cycle_max = kernel_data.query('Iteration == @iter')['End'].max()

	# Derive tick positions of kernel axis which will be middle of each kernel execution
	kernel_mids = ((kernel_data.query('Iteration == @iter')['Start'] + kernel_data.query('Iteration == @iter')['End'])/2).apply(int).get_values()

	# Derive the tick positions of page numbers x-axis which will be starting and ending of managed allocations
	cycle_bounds = np.unique(np.append(kernel_data.query('Iteration == @iter')['Start'].values, kernel_data.query('Iteration == @iter')['End'].values))

	# Create figure for histogram 
	fig = plt.figure(figsize=(10,5))

	# Set font of the plot
	font = {'family' : 'sans-serif',
        	'weight' : 'normal',
        	'size'   : 14}
	plt.rc('font', **font)

	# Set axes labels
	plt.xlabel('Time (cycle)')
	plt.ylabel('Page Number')

	# Create the primary x-axis
	prim_ax = fig.add_subplot(111)

	# Set the bounds, ticks, tick labels, and vertical grids for primary x-axis
	prim_ax.set_ylim([page_min, page_max]) 
	prim_ax.set_yticks(managed_bounds)
	prim_ax.set_yticklabels(managed_bounds, rotation=0)

	prim_ax.set_xticks(cycle_bounds)
	prim_ax.set_xticklabels(cycle_bounds, rotation=45)

	prim_ax.grid(b=True, which='major', color='grey', linestyle='-')

	# Query RD field to color Read/Write access and check for empty query output
	# Check for empty query values
	if not acc_data.query('RD == 0').empty:
		plt.scatter(acc_data.query('RD == 0 & Cycle >= @cycle_min & Cycle <= @cycle_max')['Cycle'].values, acc_data.query('RD == 0 & Cycle >= @cycle_min & Cycle <= @cycle_max')['PageNum'].values, marker='o', edgecolor='none', s=1, c='k', label='Write')

	if not acc_data.query('RD == 1').empty:
		plt.scatter(acc_data.query('RD == 1 & Cycle >= @cycle_min & Cycle <= @cycle_max')['Cycle'].values, acc_data.query('RD == 1 & Cycle >= @cycle_min & Cycle <= @cycle_max')['PageNum'].values, marker='x', edgecolor='none', s=1, c='r', label='Read')

	# Create the secondary y-axis
	sec_y_ax = prim_ax.twinx()

	# Set the bounds, ticks, tick labels for secondary y-axis
	sec_y_ax.set_ylim([page_min, page_max])
	sec_y_ax.set_yticks(alloc_mids)
	sec_y_ax.set_yticklabels(alloc_names, rotation=0)

	# Create the secondary x-axis
	sec_x_ax = prim_ax.twiny()

	# Set the bounds, ticks, tick labels for secondary x-axis
	sec_x_ax.set_xlim(prim_ax.get_xlim())
	sec_x_ax.set_xticks(kernel_mids)
	sec_x_ax.set_xticklabels(kernel_names)

	# Set the legends for plot
	prim_ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), prop={'size': 14})
 
	plt.savefig(acc_dir + benchmark + '_iter' + str(iter) + '.png',  dpi=300, bbox_inches="tight")

