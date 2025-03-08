import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpmath as mp
###############################################################################
# 1) LOSS AND HELPER FUNCTIONS
###############################################################################
def loss(w1, w2):
    return (w1 * w2 - 5) ** 2
def gradient_mpf(w1, w2):
    # Gradient of (w1*w2 - 5)^2
    return (w1*w2 - mp.mpf('5')) * w2, (w1*w2 - mp.mpf('5')) * w1
def gradient_descent_mp(w_init, eta, steps):
    """High-precision Gradient Descent."""
    w1, w2 = mp.mpf(w_init[0]), mp.mpf(w_init[1])
    traj = [(w1, w2)]
    balancing_metric = []
    for _ in range(steps):
        gw1, gw2 = gradient_mpf(w1, w2)
        w1 -= eta * gw1
        w2 -= eta * gw2
        traj.append((w1, w2))
        # Our balancing metric: |sigma1^2 - sigma2^2|
        diff_squares = mp.fabs(w1**2 - w2**2)
        balancing_metric.append(diff_squares)
    return traj, balancing_metric
###############################################################################
# 2) SET UP CONTOUR PLOT
###############################################################################
# Fine grid for contour
w1_vals = np.linspace(0, 3, 200)
w2_vals = np.linspace(0, 3, 200)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Loss = loss(W1, W2)
plt.figure(figsize=(8, 6))
plt.contourf(W1, W2, Loss, levels=100, cmap=plt.cm.Greys, alpha=1.0)
plt.colorbar()
###############################################################################
# 3) PLOT BALANCED SOLUTION AND XY=5
###############################################################################
sqrt5 = np.sqrt(5)
plt.scatter(sqrt5, sqrt5, color='black', s=100, marker='x',
            label=r'Balanced $(\sqrt{5}, \sqrt{5})$', linewidth=3)
# Dashed line for w1*w2=5
w1_line = np.linspace(0.1, 4, 300)
w2_line = 5 / w1_line
plt.plot(w1_line, w2_line, 'k--', label=r'$\sigma_1 \cdot \sigma_2 = 5$', alpha=0.5)
###############################################################################
# 4) GF TRAJECTORY: x^2 - y^2 = constant
###############################################################################
w_init_GF = [1.5, 2.25]
C = w_init_GF[0]**2 - w_init_GF[1]**2  # x0^2 - y0^2
# Intersection with xy=5 => solve x^2 - (5/x)^2 = C => x^4 - C*x^2 - 25 = 0
discriminant = np.sqrt(C**2 + 100)
u = (C + discriminant) / 2  # positive root
x_int = np.sqrt(u)
y_int = 5 / x_int
# Parametric GF trajectory from initial x to intersection
gf_x = np.linspace(w_init_GF[0], x_int, 100)
gf_y = np.sqrt(gf_x**2 - C)
# Plot GF path
plt.plot(gf_x, gf_y, color='red', linestyle='-', linewidth=2, label='GF Trajectory')
plt.scatter(w_init_GF[0], w_init_GF[1], color='blue', s=100, marker='x',
            label='GF Initial', linewidth=3)
###############################################################################
# 5) RUN GD FOR THREE LEARNING RATES & PLOT ON SAME FIGURE
###############################################################################
lrs = [mp.mpf('0.18'), mp.mpf('0.1997'), mp.mpf('0.21')]
colors = ['blue', 'red', 'green']
labels = ['Stable GD', 'GD at EOS', 'GD Beyond EOS']
gd_init = [1.5, 2.25]  # same init as GF
steps = 500
# We'll store balancing metrics for each LR for the second figure
gd_balance_dict = {}
for lr, c in zip(lrs, colors):
    traj, balance = gradient_descent_mp(gd_init, lr, steps)
    # Convert to float for plotting
    traj_np = np.array([[float(t[0]), float(t[1])] for t in traj])
    # Plot the GD trajectory
    lr_str = str(lr)
    plt.plot(traj_np[:, 0], traj_np[:, 1], color=c, linewidth=2,
             label=f'GD (LR={lr_str})')
    # Mark the initial point (same as gf_init)
    plt.scatter(traj_np[0, 0], traj_np[0, 1], color=c, s=100, marker='o')
    # Store the balancing metric for the second figure
    gd_balance_dict[lr_str] = balance
plt.xlabel(r'$\sigma_1$', fontsize=17)
plt.ylabel(r'$\sigma_2$', fontsize=17)
plt.xlim(1.25, 3)
plt.ylim(1.6, 3)
plt.title("GF + GD Trajectories on Loss Contour", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'gd_trajlrs.png', dpi=300, bbox_inches='tight')
plt.show()
###############################################################################
# 6) SECOND FIGURE: PLOT BALANCING METRIC (|sigma1^2 - sigma2^2|)
###############################################################################
plt.figure(figsize=(7.5, 5))
# First, the GF balancing metric is constant along x^2 - y^2 = C.
# For the chosen init (1.5, 2.25), C = 1.5^2 - 2.25^2 = -2.8125 => abs = 2.8125
gf_balancing = abs(w_init_GF[0]**2 - w_init_GF[1]**2)
plt.axhline(gf_balancing, color='black', linestyle='--', linewidth=3,
            label=f'GF (Constant)')
counter = 0
for lr, c in zip(lrs, labels):
    lr_str = str(lr)
    # Convert from mp to float
    bal_float = [float(val) for val in gd_balance_dict[lr_str]]
    plt.plot(bal_float, linewidth=2, label=f'{c}')
plt.xlabel("Iteration", fontsize=17)
plt.ylabel(r"Balancing Gap", fontsize=17)
#plt.title("Balancing gap", fontsize=14)
plt.yscale('log')  # Helps visualize small changes
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.savefig(f'Desktop/gd_balancing_diff.pdf', dpi=300, bbox_inches='tight')
plt.show()