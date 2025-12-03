import matplotlib.pyplot as plt
import numpy as np

def draw_autoencoder_corrected():
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # Layer coordinates
    layers_x = [1, 4, 7, 10, 13]
    layer_names = ['Input\n(x₁,x₂,x₃)', 'u₁,u₂\n(Hidden 1)', 'y₁,y₂\n(ReLU)', 'u₃,u₄,u₅\n(Hidden 2)', 'Output\n(x̂₁,x̂₂,x̂₃)']
    
    # Node positions
    nodes = {
        'x1': (layers_x[0], 2.5), 'x2': (layers_x[0], 1.5), 'x3': (layers_x[0], 0.5),
        'u1': (layers_x[1], 2), 'u2': (layers_x[1], 1),
        'y1': (layers_x[2], 2), 'y2': (layers_x[2], 1),
        'u3': (layers_x[3], 2.5), 'u4': (layers_x[3], 1.5), 'u5': (layers_x[3], 0.5),
        'o1': (layers_x[4], 2.5), 'o2': (layers_x[4], 1.5), 'o3': (layers_x[4], 0.5)
    }
    
    # Draw connections
    connections = [
        # x to u layer
        ('x1', 'u1', 'w₁=+1'), ('x1', 'u2', 'w₂=+1'),
        ('x2', 'u1', 'w₃=+1'), ('x2', 'u2', 'w₄=-1'),
        ('x3', 'u1', 'w₅=+1'), ('x3', 'u2', 'w₆=+1'),
        
        # u to y (ReLU)
        ('u1', 'y1', 'σ'), ('u2', 'y2', 'σ'),
        
        # y to u3,u4,u5
        ('y1', 'u3', 'w₇=-1'), ('y1', 'u4', 'w₈=+1'), ('y1', 'u5', 'w₉=+1'),
        ('y2', 'u3', 'w₁₀=+1'), ('y2', 'u4', 'w₁₁=-1'), ('y2', 'u5', 'w₁₂=+1'),
        
        # u to output (ReLU)
        ('u3', 'o1', 'σ'), ('u4', 'o2', 'σ'), ('u5', 'o3', 'σ')
    ]
    
    # Draw connections
    for start, end, label in connections:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.6, linewidth=1)
        
        # Add weight label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.15
        if abs(y1 - y2) < 0.5:  # nearly horizontal
            ax.text(mid_x, mid_y + offset, label, ha='center', va='bottom', 
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
        else:
            ax.text(mid_x, mid_y, label, ha='center', va='center', 
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # Node values and gradients
    node_data = {
        'x1': ('x₁ = 1', ''), 'x2': ('x₂ = -1', ''), 'x3': ('x₃ = 1', ''),
        'u1': ('u₁ = 1', ''), 'u2': ('u₂ = 3', ''),
        'y1': ('y₁ = 1', '∂ℓ/∂y₁ = 4'), 'y2': ('y₂ = 3', '∂ℓ/∂y₂ = 8'),
        'u3': ('u₃ = 2', '∂ℓ/∂u₃ = 2'), 'u4': ('u₄ = -2', '∂ℓ/∂u₄ = 0'), 'u5': ('u₅ = 4', '∂ℓ/∂u₅ = 6'),
        'o1': ('x̂₁ = 2', ''), 'o2': ('x̂₂ = 0', ''), 'o3': ('x̂₃ = 4', '')
    }
    
    # Draw nodes
    for node, (value, grad) in node_data.items():
        x, y = nodes[node]
        
        # Node circle
        circle = plt.Circle((x, y), 0.25, fill=True, facecolor='lightblue', 
                          edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        
        # Value
        ax.text(x, y + 0.1, value, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Gradient in red
        if grad:
            ax.text(x, y - 0.15, grad, ha='center', va='center', fontsize=9, 
                   color='red', fontweight='bold')
    
    # Weight gradients (shown near weights)
    weight_grads = {
        'w₇': 2, 'w₈': 0, 'w₉': 6,
        'w₁₀': 6, 'w₁₁': 0, 'w₁₂': 18
    }
    
    # Mark some key weight gradients
    grad_positions = [
        (5.5, 2.1, '∂ℓ/∂w₇ = 2'), (5.5, 1.6, '∂ℓ/∂w₈ = 0'), (5.5, 1.1, '∂ℓ/∂w₉ = 6'),
        (6.0, 2.0, '∂ℓ/∂w₁₀ = 6'), (6.0, 1.5, '∂ℓ/∂w₁₁ = 0'), (6.0, 1.0, '∂ℓ/∂w₁₂ = 18')
    ]
    
    for x, y, text in grad_positions:
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color='darkred',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="mistyrose"))
    
    # Layer labels
    for i, (x, name) in enumerate(zip(layers_x, layer_names)):
        ax.text(x, 3.2, name, ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor='gray'))
    
    # Loss info
    ax.text(14.5, 1.5, 'Loss: ℓ = 11\n(2-1)² + (0-(-1))² + (4-1)²\n= 1 + 1 + 9 = 11', 
           ha='left', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # Setup
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('Autoencoder Architecture with Backpropagation Gradients\n(Red: ∂ℓ/∂variable, Dark red: ∂ℓ/∂weight)', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax

fig, ax = draw_autoencoder_corrected()
plt.show()