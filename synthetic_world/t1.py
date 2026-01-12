import numpy as np
import matplotlib.pyplot as plt

x = np.load("./synthetic_world_debug/temporal_gt.npy")

plt.plot(x)
plt.ylim(-0.1, 1.1)
plt.title("Temporal Ground Truth")
plt.show()

import numpy as np

x = np.load("./synthetic_world_debug/temporal_gt.npy")

print(x.shape)
print(x[:120])
print("num positive frames:", x.sum())

