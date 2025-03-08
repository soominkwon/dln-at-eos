import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpmath as mp

def loss(w1, w2):
    return (w1 * w2 - 5) ** 2
w1_vals = np.linspace(0, 3, 200)
w2_vals = np.linspace(0, 3, 200)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Loss = loss(W1, W2)
# Set up learning rate (mpmath) for naming
eta = mp.mpf('0.1')
eta_str = str(eta)  # Convert to string for file naming

# Contour plot
plt.figure(figsize=(8, 6))
plt.contourf(W1, W2, Loss, levels=100, cmap=plt.cm.Greys, alpha=1.0)
plt.colorbar()

###############################################################################
# 2) BALANCED SOLUTION & XY=5 LINE
###############################################################################
sqrt5 = np.sqrt(5)
w_init_GF = [1.5, 2.25]
C = w_init_GF[0]**2 - w_init_GF[1]**2  # x0^2 - y0^2

plt.scatter(sqrt5, sqrt5, color='black', s=100, marker='x',
            label=r'Balanced', linewidth=3)
plt.scatter(w_init_GF[0], w_init_GF[1], color='blue', s=100, marker='x',
            label='Initial', linewidth=3)
# Dashed line for w1 * w2 = 5
w1_line = np.linspace(0.1, 4, 300)
w2_line = 5 / w1_line
plt.plot(w1_line, w2_line, 'k--', label=r'$\sigma_1 \cdot \sigma_2 = 5$', alpha=0.5)
###############################################################################
# 3) GF TRAJECTORY: x^2 - y^2 = constant
###############################################################################

# Intersection with xy = 5 => solve x^2 - (5/x)^2 = C => x^4 - C*x^2 - 25 = 0
discriminant = np.sqrt(C**2 + 100)
u = (C + discriminant) / 2  # choose the positive root
x_int = np.sqrt(u)
y_int = 5 / x_int
# Generate the GF trajectory
gf_x = np.linspace(w_init_GF[0], x_int, 100)
gf_y = np.sqrt(gf_x**2 - C)
# Plot GF trajectory (red line)
plt.plot(gf_x, gf_y, color='green', linestyle='-', linewidth=2, label='GF Trajectory')
# Plot initial point for GF (blue "x")


###############################################################################
# 4) GD TRAJECTORY (HIGH PRECISION) + PLOTTING
###############################################################################
def gradient_mpf(w1, w2):
    return (w1*w2 - mp.mpf('5')) * w2, (w1*w2 - mp.mpf('5')) * w1
def gradient_descent_mp(w_init, eta, steps):
    w1, w2 = mp.mpf(w_init[0]), mp.mpf(w_init[1])
    traj = [(w1, w2)]
    balancing_metric = []
    for _ in range(steps):
        gw1, gw2 = gradient_mpf(w1, w2)
        w1 -= eta * gw1
        w2 -= eta * gw2
        traj.append((w1, w2))
        # |sigma1^2 - sigma2^2|
        diff_squares = mp.fabs(w1**2 - w2**2)
        balancing_metric.append(diff_squares)
        print(diff_squares)
    return traj, balancing_metric
# Run GD
gd_init = [1.5, 2.25]   # Example initial point
steps = 200
gd_traj, gd_balance = gradient_descent_mp(gd_init, eta, steps)


# Convert trajectory to floats for plotting
gd_traj_np = np.array([[float(x[0]), float(x[1])] for x in gd_traj])
# Plot GD trajectory
plt.plot(gd_traj_np[:, 0], gd_traj_np[:, 1], color='red', linewidth=2,
         label='GD Trajectory')
#plt.scatter(gd_traj_np[0, 0], gd_traj_np[0, 1], color='green', s=100, marker='o',
#            label='GD Initial')

###############################################################################
# 5) FINALIZE FIRST FIGURE
###############################################################################
plt.xlabel(r'$\sigma_1$', fontsize=17)
plt.ylabel(r'$\sigma_2$', fontsize=17)
plt.xlim(1.25, 3)
plt.ylim(1.6, 3)
plt.legend(fontsize=14)
# Save figure with learning rate in filename
plt.savefig(f'Desktop/contour_gf.pdf', dpi=300, bbox_inches='tight')
plt.show()

