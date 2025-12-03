import numpy as np
import matplotlib.pyplot as plt

# Define functions
def relu(t):
    return np.maximum(0, t)

def h1(x):
    return relu(1 - x)

def h2(x):
    return relu(1 + x)

def h(x):
    return np.maximum(2, np.abs(x) + 1)

# Create plot
x = np.linspace(-3, 3, 400)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot h1(x) = ReLU(1 - x)
ax = axes[0]
ax.plot(x, h1(x), 'b-', linewidth=2, label=r'$h_1(x) = \sigma(1 - x)$')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('h₁(x)')
ax.set_title(r'$h_1(x) = \sigma(1 - x)$')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(-0.5, 4.5)

# Plot h2(x) = ReLU(1 + x)
ax = axes[1]
ax.plot(x, h2(x), 'r-', linewidth=2, label=r'$h_2(x) = \sigma(1 + x)$')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('h₂(x)')
ax.set_title(r'$h_2(x) = \sigma(1 + x)$')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(-0.5, 4.5)

# Plot h(x) = h1(x) + h2(x)
ax = axes[2]
ax.plot(x, h(x), 'g-', linewidth=3, label=r'$h(x) = \max\{2, |x|+1\}$')
ax.plot(x, h1(x) + h2(x), 'k--', linewidth=1.5, alpha=0.7, label='h₁(x)+h₂(x)')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
# Mark key points
ax.plot([-1, -1], [0, 2], 'k:', alpha=0.5)
ax.plot([1, 1], [0, 2], 'k:', alpha=0.5)
ax.plot(-1, 2, 'ro', markersize=8)
ax.plot(1, 2, 'ro', markersize=8)
ax.text(-1.2, 2.2, '(-1, 2)', fontsize=9)
ax.text(1.05, 2.2, '(1, 2)', fontsize=9)
ax.set_xlabel('x')
ax.set_ylabel('h(x)')
ax.set_title(r'$h(x) = \max\{2, |x|+1\} = h_1(x) + h_2(x)$')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(-0.5, 5)

plt.tight_layout()
plt.show()