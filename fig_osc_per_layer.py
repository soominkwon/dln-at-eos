#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:05:38 2024

@author: soominkwon
"""

from mpmath import mp
import matplotlib.pyplot as plt

# Set precision (e.g., 50 decimal places)
mp.dps = 70
M = mp.randmatrix(5) 

U, S, V = mp.svd_r(M)

new_S = mp.diag([10, 8, 6, 0, 0])
M = new_S 

# Objective function and its gradient
def objective(x, y, z):
    diff = (x @ y @ z - M)
    return mp.norm(diff)**2

def gradient(x, y, z):
    # Note that the gradients do not have a factor of 2 in them
    grad_x = 1 * (x @ y @ z - M)@z.T @ y.T
    grad_y = 1 * x.T@(x @ y @ z - M) @ z.T
    grad_z = 1 * y.T@x.T@(x @ y @ z - M)
    return grad_x, grad_y, grad_z

# Parameters
eta = mp.mpf('0.010') 
alpha = mp.mpf('0.03') # Step size

x = mp.zeros(5)  # Initial x
y = alpha*mp.eye(5)
z = alpha*mp.eye(5)
num_iterations = 2500 # Number of iterations


# Lists to store trajectory, |x^2 - y^2|, and objective
trajectory_x = [x]
trajectory_y = [y]
trajectory_z = [z]

abs_diff1 = []
abs_diff2 = []
abs_diff3 = []

objective_vals = [objective(x, y, z)]
# Gradient descent loop
for t in range(num_iterations):
    grad_x, grad_y, grad_z = gradient(x, y, z)
    x -= eta * grad_x
    y -= eta * grad_y
    z -= eta * grad_z

    trajectory_x.append(x)
    trajectory_y.append(y)
    trajectory_z.append(z)

    abs_diff1.append(mp.fabs(x[0, 0] - y[0, 0]))
    abs_diff2.append(mp.fabs(x[1, 1] - y[1, 1]))
    abs_diff3.append(mp.fabs(x[2, 2] - y[2, 2]))

    objective_vals.append(objective(x, y, z))
    print(t, objective(x, y, z))

# Convert to floats for plotting
abs_diff1 = [float(v) for v in abs_diff1]
abs_diff2 = [float(v) for v in abs_diff2]

temp = [mp.svd_r(trajectory_x[i]@trajectory_y[i]@trajectory_z[i], compute_uv=False)[0] for i in range(num_iterations)]

objective_vals = [float(v) for v in objective_vals]

# Plotting
osc_idx = 9
eta = '040'
size = (7.5, 5)
plt.figure(figsize=size)

plt.subplot(1, 1, 1)  # 2x2 grid, first subplot
plt.plot(objective_vals[::osc_idx], label='Train Loss', linewidth=2)
#plt.ylabel('Train Loss', fontsize=17)
#plt.ylabel('', fontsize=17)
plt.legend(fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Loss', fontsize=17)


plt.savefig(f'Desktop/train_loss_lr_{eta}.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=size)

plt.subplot(1, 1, 1)  # 2x2 grid, first subplot
#plt.plot(objective_vals[::osc_idx], label='Train Loss', linewidth=2)
plt.plot([(trajectory_x[i])[0, 0] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_1$ (Layer 3)', color='blue', linewidth=2)
plt.plot([(trajectory_x[i])[1, 1] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_2$', color='red', linewidth=2)
plt.plot([(trajectory_x[i])[2, 2] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_3$', color='green', linewidth=2)#plt.xlabel('Iteration (x80)', fontsize=17)
#plt.ylabel('Train Loss', fontsize=17)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Magnitude', fontsize=15)
#plt.ylabel('', fontsize=17)
plt.legend(fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)


plt.savefig(f'Desktop/layer3_lr_{eta}.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=size)

plt.subplot(1, 1, 1)  # 2x2 grid, first subplot
#plt.plot(objective_vals[::osc_idx], label='Train Loss', linewidth=2)
plt.plot([(trajectory_y[i])[0, 0] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_1$ (Layer 2)', color='blue', linewidth=2)
plt.plot([(trajectory_y[i])[1, 1] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_2$', color='red', linewidth=2)
plt.plot([(trajectory_y[i])[2, 2] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_3$', color='green', linewidth=2)#plt.xlabel('Iteration (x80)', fontsize=17)
#plt.ylabel('Train Loss', fontsize=17)
#plt.ylabel('', fontsize=17)
plt.legend(fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Magnitude', fontsize=15)

plt.savefig(f'Desktop/layer2_lr_{eta}.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=size)

plt.subplot(1, 1, 1)  # 2x2 grid, first subplot
#plt.plot(objective_vals[::osc_idx], label='Train Loss', linewidth=2)
plt.plot([(trajectory_z[i])[0, 0] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_1$ (Layer 1)', color='blue', linewidth=2)
plt.plot([(trajectory_z[i])[1, 1] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_2$', color='red', linewidth=2)
plt.plot([(trajectory_z[i])[2, 2] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_3$', color='green', linewidth=2)#plt.xlabel('Iteration (x80)', fontsize=17)
#plt.ylabel('Train Loss', fontsize=17)
#plt.ylabel('', fontsize=17)
plt.legend(fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Magnitude', fontsize=15)

plt.savefig(f'Desktop/layer1_lr_{eta}.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=size)

plt.subplot(1, 1, 1)  # 2x2 grid, first subplot
#plt.plot(objective_vals[::osc_idx], label='Train Loss', linewidth=2)
plt.plot([(trajectory_x[i])[0, 0]*(trajectory_y[i])[0, 0]*(trajectory_z[i])[0, 0] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_1$', color='blue', linewidth=2)
plt.plot([(trajectory_x[i])[1, 1]*(trajectory_y[i])[1, 1]*(trajectory_z[i])[1, 1]for i in range(num_iterations)][::osc_idx], label=r'$\sigma_2$', color='red', linewidth=2)
plt.plot([(trajectory_x[i])[2, 2]*(trajectory_y[i])[2, 2]*(trajectory_z[i])[2,2] for i in range(num_iterations)][::osc_idx], label=r'$\sigma_3$', color='green', linewidth=2)#plt.xlabel('Iteration (x80)', fontsize=17)
#plt.ylabel('Train Loss', fontsize=17)
#plt.ylabel('', fontsize=17)
plt.legend(fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Magnitude', fontsize=15)

plt.savefig(f'Desktop/end_to_end_lr_{eta}.pdf', dpi=300, bbox_inches='tight')
plt.show()