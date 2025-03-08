import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to compute the loss
def loss(w1, w2):
    return  (w1 * w2 - 5) ** 2  # Loss centered at w1*w2 = 5

# Gradient of the loss function
def gradient(w1, w2):
    grad_w1 = 1*(w1 * w2 - 5) * w2
    grad_w2 = 1*(w1 * w2 - 5) * w1
    #grad_w1 = (w2**2 + 2*w1*w2 - 5)/np.sqrt(2) + 0.5*0.5**2 *(w1 + 2*w2 - 5) # This is second term + third term
    #grad_w2 = (w1**2 + 2*w1*w2 - 5)/np.sqrt(2) + 0.5*0.5**2 *(2*w1 + w2 - 5) # This is second term + third term

    return np.array([grad_w1, grad_w2])

# Gradient Descent function
def gradient_descent(w_init, eta, steps):
    w = np.array(w_init)
    trajectory = [w.copy()]
    losses = []
    prods = []
    for i in range(steps):
        grad = gradient(w[0], w[1])

        #if i > 4000:
        #    eta = 0.1
        w -= eta * grad
        trajectory.append(w.copy())
        losses.append(loss(w[0], w[1]))
        print(w[0])
        prods.append(w[0]*w[1])
    return np.array(trajectory), np.array(losses), np.array(prods)

# Grid setup
w1 = np.linspace(0, 3, 100)  # Extend range for clarity
w2 = np.linspace(0, 3, 100)
W1, W2 = np.meshgrid(w1, w2)
Loss = loss(W1, W2)
cmap = plt.cm.Greys  # Using the 'Greys' colormap
norm = mcolors.Normalize(vmin=0, vmax=np.max(Loss))


# Dotted line for w1 * w2 = 5
w1_line = np.linspace(0.1, 4, 300)
w2_line = 5 / w1_line

# Plot configuration
plt.figure(figsize=(8, 6))
plt.contourf(W1, W2, Loss, levels=100, cmap=cmap, alpha=1.0)
plt.colorbar()

# Dotted line for w1 * w2 = 5

# Marking the purple "X" at (\sqrt{5}, \sqrt{5})
sqrt5 = np.sqrt(5)
plt.scatter(sqrt5, sqrt5, color='black', s=100, marker='x', label=r'Balanced', linewidth=3)

# GD Trajectory
#w_init = [2.0833333,2.4]  # Initialization
w_init = [1.5, 2.25]
#val = np.sqrt(5) + 0.5 * (1/np.sqrt(2))
#w_init = [val , val]

eta = 0.1997  # Learning rate
steps = 5000  # Number of iterations
trajectory, losses, prod = gradient_descent(w_init, eta, steps)
plt.scatter(trajectory[0, 0],trajectory[0, 1], color='blue', s=100, marker='x', label=r'Initial', linewidth=3)
plt.plot(w1_line, w2_line, 'k--', label=r'$\sigma_1 \cdot \sigma_2 = 5$', alpha=0.5,)
plt.plot(trajectory[::31, 0], trajectory[::31, 1], color='red', label="GD Trajectory", linewidth=1.0)

# Annotations
# plt.title(r'Initialization: $w_1=2.8$, $w_2=3.4$, $\eta=10^{-2}$')
plt.xlabel(r'$\sigma_1$', fontsize=17)
plt.ylabel(r'$\sigma_2$', fontsize=17)
# Zoom into the region of interest
#plt.xlim(1, 4)
#plt.ylim(1, 4)
plt.xlim(1.25, 3)
plt.ylim(1.6, 3)
plt.legend(fontsize=15)
plt.savefig(f'Desktop/contour_gd_below_stab.pdf', dpi=300, bbox_inches='tight')

# Show plot
plt.show()
