import math
from tkinter import YView

import numpy as np

import random
import seaborn as sn

from torch import ao


def sgn(x):
    return -1.0 if x < 0.0 else 1.0


reciprocal = lambda a: 1 / a if abs(a) > 0.05 else 20.0 * sgn(a)
exp = lambda a: math.exp(min(a, 10.0))
trunc = lambda a: float(int(a))
#abs = lambda a: abs(a)
sin = lambda a: math.sin(a)
sqrt = lambda a: math.sqrt(a) if a >= 0.0 else 0.0
cos = lambda a: math.cos(a)
neg = lambda a: -a

#min = lambda a, b: min(a, b)
#max = lambda a, b: max(a, b)

select = lambda a, iftrue, iffalse: iftrue if a > 0 else iffalse

x = [1, 1, 0, 0]

#cos(x[1]) + x[0]
#t = reciprocal(abs((cos(x[1]) + x[0])))


# -------------------


#f0 = lambda x:  min(reciprocal(abs((cos(x[1]) + x[0]))), x[1])
#f1 = lambda x: -exp(trunc(max((x[0] if -abs(x[1]) > 0 else x[1]), x[0])))

#f0 = lambda x: exp((x[1] if cos(-abs(-5.622074044580325)) > 0 else x[0]))
#f1 = lambda x: (-exp(-sqrt(sin(abs(x[1])))) + x[0])


# Diagonal programs

"""
# 1
tit = 'prog_1'
f0 = lambda x: -exp(max(-sin((x[0] if trunc(x[1]) > 0 else x[1])), x[0]))
f1 = lambda x: -cos(((((x[1] + x[0]) + x[1]) * x[0]) * x[1]))

#2
tit = 'prog_2'
f0 = lambda x: (-cos(cos(x[1])) if abs(-66.31885466661134) > 0 else x[0])
f1 = lambda x: -abs(cos(max(cos(-sqrt(x[1])), x[0])))

#3
tit = 'prog_3'
f0 = lambda x: -exp(max((max(-abs(x[1]), x[0]) * x[1]), x[0]))
f1 = lambda x: -abs(reciprocal(-sqrt(((x[1] + x[0]) * x[1]))))

#4
tit = 'prog_4'
f0 = lambda x: (-cos(cos(x[1])) if exp(-64.18861262866074) > 0 else x[0])
f1 = lambda x: neg(cos(max(cos(-sqrt(x[1])), x[0])))
"""

tit = 'prog_controls_1'
f0 = lambda x: -sin(((x[1] if (x[3] * x[2]) > 0 else x[0])) + 0.3)
#f0 = lambda x: 0
f1 = lambda x: ((x[0] if (x[2] if sin(-sin(x[3])) > 0 else x[1]) > 0 else x[3])) - 0.3

#---------------------


#a[1] = -exp(trunc(max((x[0] if -abs(x[1]) > 0 else x[1]), x[0])))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal', adjustable='box')

square = patches.Rectangle((0, 0), 0.1, 0.1, facecolor='green')
ax.add_patch(square)
square = patches.Rectangle((0.4, 0.4), 0.2, 0.2, facecolor='red')
ax.add_patch(square)

X, Y, U, V = [], [], [], []

SPEED_X, SPEED_Y = -1, -1

for x in np.arange(0, 1, 0.1):
    for y in np.arange(0, 1, 0.1):
        i = [x, y, SPEED_X, SPEED_Y]
        a0 = f0(i)
        a1 = f1(i)

        X.append(x)
        Y.append(y)
        U.append(a0)
        V.append(a1)

plt.quiver(X, Y, U, V, color='blue', units='xy')
#plt.scatter(X, Y, s=U, color='blue')




# Plotting Vector Field with QUIVER

plt.title(tit)

# Setting x, y boundary limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# Show plot with grid
plt.grid()

plt.savefig(f'{tit}.pdf')

plt.show()


print('done')
