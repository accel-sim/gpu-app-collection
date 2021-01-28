import os

benchmarks = ['backprop','bfs','fdtd','hotspot','needle','ra','srad','sssp']
for b in benchmarks:
	print('Plotting benchmark: ' + b)
	os.system('python plot_access_individual.py ' + b)
