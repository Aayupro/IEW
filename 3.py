import lasio
import numpy as np
import matplotlib.pyplot as plt

lasfile = "/Users/aayushmishra/Documents/Projects/IEW/F02-1_logs.las"
las = lasio.read(lasfile)

print("Available logs:", las.curves.keys())

# Use correct curve names
depth = las['DEPTH']
GR = las['GR']
# Shale has high GR (>75 API), sand has low GR (<75 API)
# Hydrocarbon zones are typically in low GR intervals
hydrocarbon_mask = GR < 75  # True where sand/hydrocarbon likely

# Plot to visualize
plt.figure(figsize=(5,10))
plt.plot(GR, depth, color='green')
plt.gca().invert_yaxis()  # depth increases downwards
plt.fill_betweenx(depth, GR, 75, where=hydrocarbon_mask, color='red', alpha=0.3)
plt.xlabel('Gamma Ray (API)')
plt.ylabel('Depth (m)')
plt.title('Hydrocarbon Zones (red)')
plt.show()



# Assuming seismic cube sampling rate is in meters per sample
z_start = f.samples[0]           # top of seismic cube in depth
z_interval = f.samples[1] - f.samples[0]  # depth per sample

# Convert depth to seismic sample index
sample_indices = ((depth - z_start) / z_interval).astype(int)
sample_indices = np.clip(sample_indices, 0, cube.shape[2]-1)

# Now hydrocarbon mask can be applied to seismic cube

# Initialize label cube same shape as seismic cube
labels = np.zeros_like(cube, dtype=int)  # 0 = non-hydrocarbon

# Map hydrocarbon zones to inline/crossline where well is located
well_i = 100  # example inline index of well
well_j = 200  # example crossline index of well

# Fill label cube along well path
labels[well_i, well_j, sample_indices[hydrocarbon_mask]] = 1  # 1 = hydrocarbon
