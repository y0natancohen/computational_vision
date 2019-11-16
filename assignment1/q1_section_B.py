import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
# Avi's position
z_w = np.arange(100, 1000) # The z coordinate of avi moves in the range of 100 to 1000.
p0w_x = np.zeros_like(z_w)
p1w_x = np.full_like(z_w, 200)
# Camera world position
c_x = 70
c_z = -50
# Camerea focal length
f = 10
# =================  Change these to the solution of the question
x0_i_list = np.zeros_like(z_w, dtype=float)  # len is 900
x1_i_list = np.zeros_like(z_w, dtype=float)
# =================

# ======== my code here ===========
# put camera in (0,0)
c_x_normalized = 0
c_z_normalized = 0
p0w_x_normalized = p0w_x - c_x
p1w_x_normalized = p1w_x - c_x
z_w_normalized = z_w - c_z
projection_z = f
for i, coordinate in enumerate(z_w_normalized):
    xw0 = p0w_x_normalized[i]
    xw1 = p1w_x_normalized[i]
    zw = z_w_normalized[i]
    x0_i = xw0
    x1_i = xw1
    x0_i_list[i] = x0_i
    x1_i_list[i] = x1_i

# ======== my code here ===========


plt.plot(z_w, x0_i_list, color='blue')
plt.plot(z_w, x1_i_list, color='green')
plt.title("avi's parallel  projections")
plt.show()
plt.savefig("avi_parallel.png")

