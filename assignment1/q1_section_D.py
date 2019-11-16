import matplotlib.pyplot as plt
import numpy as np

z_w = np.arange(100, 1000, 0.001)
# Camerea focal length
f = 10
# Camera world position
c_x = 70
c_z = -50


# =================  Change these to the solution of the question
x_w = np.sin((z_w-100)/100.0*2*np.pi)*10
y_w = (np.cos((z_w-100)/100.0*2*np.pi) * 10) + 190

# TODO: projection or parallel?
x_i = []
y_i = []

#=====================

_, ax = plt.subplots(figsize=(10, 10))
ax.plot(y_i, x_i)
plt.title("avi's crazy")
plt.show()
