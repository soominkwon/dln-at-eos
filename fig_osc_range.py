#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:05:38 2024

@author: soominkwon
"""

from mpmath import mp
import matplotlib.pyplot as plt

# Set precision (e.g., 50 decimal places)
mp.dps = 50
M = mp.randmatrix(5) 

U, S, V = mp.svd_r(M)

new_S = mp.diag([10, 9.5, 9, 0, 0])
M = new_S

# Objective function and its gradient for DMF with singular vector stationarity
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
start = 0.0305
end = 0.03630
increment = 0.00025

etas = [round(start + i * increment, 7) for i in range(int((end - start) / increment) + 1)]

num_iterations = 1500 # Number of iterations

alpha = mp.mpf('0.01') # Step size

osc_range1 = []
osc_range2 = []
osc_range3 = []

# Gradient descent loop
for eta in etas:
    x = mp.zeros(5)  # Initial x
    # x = alpha*mp.eye(5)
    y = alpha*mp.eye(5)
    z = alpha*mp.eye(5)

    # Lists to store trajectory, |x^2 - y^2|, and objective
    trajectory_x = [x]
    trajectory_y = [y]
    trajectory_z = [z]

    abs_diff1 = []
    abs_diff2 = []

    objective_vals = [objective(x, y, z)]

    for t in range(num_iterations):
        grad_x, grad_y, grad_z = gradient(x, y, z)
        x -= eta * grad_x
        y -= eta * grad_y
        z -= eta * grad_z

        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_z.append(z)

        abs_diff1.append(mp.fabs(mp.svd_r(x, compute_uv=False)[0] - mp.svd_r(y, compute_uv=False)[0]))
        abs_diff2.append(mp.fabs(mp.svd_r(x, compute_uv=False)[1] - mp.svd_r(y, compute_uv=False)[1]))

        objective_vals.append(objective(x, y, z))
        print(t, objective(x, y, z))

    temp1 = [(trajectory_x[i]@trajectory_y[i]@trajectory_z[i])[0, 0] for i in range(num_iterations)]
    temp2 = [(trajectory_x[i]@trajectory_y[i]@trajectory_z[i])[1, 1] for i in range(num_iterations)]
    temp3 = [(trajectory_x[i]@trajectory_y[i]@trajectory_z[i])[2, 2] for i in range(num_iterations)]

    osc_range1.append(mp.fabs(temp1[-1]-temp1[-2]))
    osc_range2.append(mp.fabs(temp2[-1]-temp2[-2]))
    osc_range3.append(mp.fabs(temp3[-1]-temp3[-2]))


# Convert to floats for plotting
abs_diff1 = [float(v) for v in abs_diff1]
abs_diff2 = [float(v) for v in abs_diff2]

objective_vals = [float(v) for v in objective_vals]
# Plot |x^2 - y^2| and the objective
plt.figure(figsize=(12, 6))
# |x^2 - y^2| (log-log plot)
plt.loglog(range(len(abs_diff1)), abs_diff1, label=r'$first$', color='blue')
plt.loglog(range(len(abs_diff2)), abs_diff2, label=r'$second$', color='red')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel(r'$|x^2 - y^2|$', fontsize=14)
plt.title(r'Log-log of $|x^2 - y^2|$ over iterations', fontsize=16)
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend(fontsize=12)
plt.show()


# Objective
plt.figure(figsize=(6, 4))
plt.scatter(etas, osc_range1, label=r'$\sigma_1$', color='blue', linewidth=2)
plt.scatter(etas, osc_range2, label=r'$\sigma_2$', color='red', linewidth=2)
plt.scatter(etas, osc_range3, label=r'$\sigma_3$', color='green', linewidth=2)

plt.xlabel('Learning Rate', fontsize=17)
plt.ylabel('Oscillation Range', fontsize=17)
plt.legend(fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.axvline(x=0.03094, color='gray', linestyle='--')
plt.axvline(x=0.033134, color='gray', linestyle='--')
plt.axvline(x=0.035611, color='gray', linestyle='--')

# Add a label at the top of the plot
plt.text(0.03094, max(osc_range1)+0.3, r'$2/S_1$', color='black', ha='center', va='bottom')
plt.text(0.033134, max(osc_range1)+0.3, r'$2/S_2$', color='black', ha='center', va='bottom')
plt.text(0.035611, max(osc_range1)+0.3, r'$2/S_3$', color='black', ha='center', va='bottom')

plt.savefig(f'Desktop/eos_osc_range.pdf', dpi=300, bbox_inches='tight')
plt.show()
