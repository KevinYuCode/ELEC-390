import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

jump = pd.read_csv('jacob/0.csv')
mask = (jump.iloc[:, 0] >= 0) & (jump.iloc[:, 0] <= 5)
jump = jump.loc[mask]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# ax[0].plot(jump.iloc[:, 0], jump.iloc[:, 1]) # x acceleration
# ax[0].plot(jump.iloc[:, 0], jump.iloc[:, 1].rolling(10).mean(), color='red')
# ax[0].legend(['raw', 'rolling mean'])
# ax[0].set_xlim(0, 5)
# ax[0].set_ylabel('x-acceleration (m/s^2)')
# ax[0].set_xlabel('time (s)')
# ax[0].set_title('x-acceleration of jumping')


# ax[1].plot(jump.iloc[:, 0], jump.iloc[:, 2]) # y acceleration
# ax[1].plot(jump.iloc[:, 0], jump.iloc[:, 2].rolling(10).mean(), color='red')
# ax[1].legend(['raw', 'rolling mean'])
# ax[1].set_xlim(0, 5)
# ax[1].set_ylabel('y-acceleration (m/s^2)')
# ax[1].set_xlabel('time (s)')
# ax[1].set_title('y-acceleration of jumping')

# ax[2].plot(jump.iloc[:, 0], jump.iloc[:, 3]) # z acceleration
# ax[2].plot(jump.iloc[:, 0], jump.iloc[:, 3].rolling(10).mean(), color='red')
# ax[2].legend(['raw', 'rolling mean'])
# ax[2].set_xlim(0, 5)
# ax[2].set_ylabel('z-acceleration (m/s^2)')
# ax[2].set_xlabel('time (s)')
# ax[2].set_title('z-acceleration of jumping')



walk = pd.read_csv('jacob/4.csv')
mask = (walk.iloc[:, 0] >= 0) & (walk.iloc[:, 0] <= 5)
walk = walk.loc[mask]

ax[0].plot(walk.iloc[:, 0], walk.iloc[:, 1]) # x acceleration
ax[0].plot(walk.iloc[:, 0], walk.iloc[:, 1].rolling(10).mean(), color='red')
ax[0].legend(['raw', 'rolling mean'])
ax[0].set_xlim(0, 5)
ax[0].set_ylabel('x-acceleration (m/s^2)')
ax[0].set_xlabel('time (s)')
ax[0].set_title('x-acceleration')
ax[0].set_title('x-acceleration of walking')

ax[1].plot(walk.iloc[:, 0], walk.iloc[:, 2]) # y acceleration
ax[1].plot(walk.iloc[:, 0], walk.iloc[:, 2].rolling(10).mean(), color='red')
ax[1].legend(['raw', 'rolling mean'])
ax[1].set_xlim(0, 5)
ax[1].set_ylabel('y-acceleration (m/s^2)')
ax[1].set_xlabel('time (s)')
ax[1].set_title('y-acceleration of walking')

ax[2].plot(walk.iloc[:, 0], walk.iloc[:, 3]) # z acceleration
ax[2].plot(walk.iloc[:, 0], walk.iloc[:, 3].rolling(10).mean(), color='red')
ax[2].legend(['raw', 'rolling mean'])
ax[2].set_xlim(0, 5)
ax[2].set_ylabel('z-acceleration (m/s^2)')
ax[2].set_xlabel('time (s)')
ax[2].set_title('z-acceleration of walking')



plt.show()
