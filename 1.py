import segyio
import numpy as np
import matplotlib.pyplot as plt

segyfile = "/Users/aayushmishra/Documents/Projects/IEW/f3_dataset.sgy"

# Open the file in read-only mode
f = segyio.open(segyfile, "r", ignore_geometry=True)
f.mmap()  # memory map for faster access

# Read all traces
traces = np.array([f.trace[i] for i in range(f.tracecount)])
print("Number of traces:", f.tracecount)
print("Samples per trace:", traces.shape[1])


ilines = f.attributes(segyio.TraceField.INLINE_3D)[:]
xlines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

print("Unique inlines:", len(np.unique(ilines)))
print("Unique crosslines:", len(np.unique(xlines)))


unique_ilines = np.unique(ilines)
unique_xlines = np.unique(xlines)
n_samples = traces.shape[1]

# Initialize cube with NaNs
cube = np.full((len(unique_ilines), len(unique_xlines), n_samples), np.nan)

# Create index mappings
iline_idx = {iline: i for i, iline in enumerate(unique_ilines)}
xline_idx = {xline: i for i, xline in enumerate(unique_xlines)}

# Fill cube
for t in range(f.tracecount):
    i = iline_idx[ilines[t]]
    j = xline_idx[xlines[t]]
    cube[i, j, :] = traces[t]

print("Cube shape:", cube.shape)


plt.imshow(cube[100, :, :], cmap='gray', aspect='auto')  # inline slice
plt.colorbar()
plt.show()
