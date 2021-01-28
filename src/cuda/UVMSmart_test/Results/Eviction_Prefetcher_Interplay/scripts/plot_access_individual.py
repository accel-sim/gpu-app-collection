import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

# Declare directory and file structure
benchmark = sys.argv[1]

log_dir = '../output_logs/AccessPattern/'
plot_dir = '../plots/AccessPattern/' + benchmark + '/'

if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)

access_file = log_dir + benchmark + '_access.txt'
alloc_file = log_dir + benchmark + '_managed.txt'
kernel_file = log_dir + benchmark + '_kernel.txt'

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

# Derive the bounds of page numbers axis by finding smallest and largest page numbers of all managed allocations and give space of 100 on both end
page_min = alloc_data['PageStart'].min() - 100
page_max = alloc_data['PageEnd'].max() + 100

# Derive the tick positions of page numbers x-axis which will be starting and ending of managed allocations
managed_bounds = np.unique(np.append(alloc_data['PageStart'].values, alloc_data['PageEnd'].values))

# Derive tick positions of managed data structure axis which will be middle of each managed allocations
alloc_mids = ((alloc_data['PageStart'] + alloc_data['PageEnd'])/2).apply(int).get_values()

# Labels of managed data structure axis is the name of the allocations
alloc_names = alloc_data['DataStructure'].get_values()

if len(sys.argv) > 2:
	iterations = sys.argv[2:]
else:
	iterations = kernel_data['Iteration'].unique()

# Loop on kernel iterations to plot access pattern
for iter in iterations:
	iter = int(iter)

	cycle_min = kernel_data.query('Iteration == @iter')['Start'].min()
	cycle_max = kernel_data.query('Iteration == @iter')['End'].max()

	# Labels of kernel axis is the name of the kernels
	kernel_names = kernel_data.query('Iteration == @iter')['Kernel'].unique()

	# Derive tick positions of kernel axis which will be middle of each kernel execution
	kernel_mids = ((kernel_data.query('Iteration == @iter')['Start'] + kernel_data.query('Iteration == @iter')['End'])/2).apply(int).get_values()

	# Derive the tick positions of page numbers x-axis which will be starting and ending of managed allocations
	cycle_bounds = np.unique(np.append(kernel_data.query('Iteration == @iter')['Start'].values, kernel_data.query('Iteration == @iter')['End'].values))

	# Create figure 
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

	prim_ax.set_xlim([cycle_min - 500, cycle_max + 500]) 
	prim_ax.set_xticks(cycle_bounds)
	prim_ax.set_xticklabels(cycle_bounds, rotation=0)

	prim_ax.grid(b=True, which='major', color='grey', linestyle='-')

	plt.scatter(acc_data.query('Cycle >= @cycle_min & Cycle <= @cycle_max')['Cycle'].values, acc_data.query('Cycle >= @cycle_min & Cycle <= @cycle_max')['PageNum'].values, marker='o', edgecolor='none', s=1, c='k')

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
	sec_x_ax.set_xticklabels(kernel_names, rotation=0)

        ##################################################################################################
	# Static part to zoom for needle; extremely hardcoded
	##################################################################################################
	from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

	all_cycles = sorted(acc_data.query('Cycle >= @cycle_min & Cycle <= @cycle_max')['Cycle'].unique())
	len_all_cycle = len(all_cycles)
	zoom_plot_xmin = all_cycles[int(len_all_cycle*0.1)]
	zoom_plot_xmax = zoom_plot_xmin + 400

	axins = zoomed_inset_axes(prim_ax, 10, loc=7)

	axins.scatter(acc_data.query('Cycle >= @cycle_min & Cycle <= @cycle_max')['Cycle'].values, acc_data.query('Cycle >= @cycle_min & Cycle <= @cycle_max')['PageNum'].values, marker='o', edgecolor='none', s=2, c='k')

	axins.set_xlim(zoom_plot_xmin, zoom_plot_xmax)
	axins.set_ylim(786850, 787050)

	plt.ticklabel_format(useOffset=False)
	axins.get_xaxis().get_major_formatter().set_scientific(False)
	axins.get_yaxis().get_major_formatter().set_scientific(False)

	#plt.yticks(visible=False)
	plt.xticks(visible=False)

	from mpl_toolkits.axes_grid1.inset_locator import mark_inset

	mark_inset(prim_ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")	
	##################################################################################################

	plt.savefig(acc_dir + benchmark + '_iter' + str(iter) + '.png',  dpi=300, bbox_inches="tight")

	plt.close()
	plt.cla()
	plt.clf()

