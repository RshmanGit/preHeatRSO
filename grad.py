import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create Map
cm = plt.get_cmap("RdYlGn")

x = np.random.rand(30)
y = np.random.rand(30)
z = np.random.rand(30)
#col = [cm(float(i)/(29)) for i in xrange(29)] # BAD!!!
col = [cm(float(i)/(30)) for i in xrange(30)]

# 2D Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y, s=10, c=col, marker='o')

# 3D Plot
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(x, y, z, s=10, c=col, marker='o')

plt.show()
