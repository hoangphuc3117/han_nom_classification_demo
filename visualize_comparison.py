"""
Visualization Comparison between DHC Model and Multi-Task Model
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_comparison_diagram():
    """Create a visual comparison between DHC and Multi-Task models"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('🏛️ Model Architecture Comparison: DHC vs Multi-Task', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # DHC Model (Left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('DHC Model (Original)', fontsize=16, fontweight='bold', pad=20)
    
    # Multi-Task Model (Right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Multi-Task Model (New)', fontsize=16, fontweight='bold', pad=20)
    
    # Common style
    box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2)
    input_style = dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green', linewidth=2)
    output_style = dict(boxstyle='round,pad=0.5', facecolor='lightcoral', edgecolor='red', linewidth=2)
    
    # DHC Model Components
    # Input
    ax1.text(5, 11, 'Input\n128×128×3', ha='center', va='center', fontsize=11, 
             bbox=input_style, fontweight='bold')
    
    # Backbone
    ax1.text(5, 9, 'ResNet50\n+ CBAM', ha='center', va='center', fontsize=11, 
             bbox=box_style, fontweight='bold')
    
    # Feature
    ax1.text(5, 7, 'Features\n2048-d', ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    # Outputs
    y_positions = [5, 3.5, 2]
    outputs = ['Main Category\n(2 classes)', 'Document Type\n(3 classes)', 'Text Direction\n(2 classes)']
    
    for y, output in zip(y_positions, outputs):
        ax1.text(5, y, output, ha='center', va='center', fontsize=9.5, 
                bbox=output_style, fontweight='bold')
    
    # Arrows
    for i in range(len(y_positions)):
        if i == 0:
            arrow = FancyArrowPatch((5, 10.5), (5, 9.5), arrowstyle='->', 
                                   mutation_scale=20, linewidth=2, color='black')
        elif i == 1:
            arrow = FancyArrowPatch((5, 8.5), (5, 7.5), arrowstyle='->', 
                                   mutation_scale=20, linewidth=2, color='black')
        else:
            # Branch arrows
            arrow1 = FancyArrowPatch((5, 6.5), (5, y_positions[0]+0.5), arrowstyle='->', 
                                    mutation_scale=15, linewidth=1.5, color='navy')
            arrow2 = FancyArrowPatch((5, 6.5), (5, y_positions[1]+0.5), arrowstyle='->', 
                                    mutation_scale=15, linewidth=1.5, color='navy')
            arrow3 = FancyArrowPatch((5, 6.5), (5, y_positions[2]+0.5), arrowstyle='->', 
                                    mutation_scale=15, linewidth=1.5, color='navy')
            ax1.add_patch(arrow1)
            ax1.add_patch(arrow2)
            ax1.add_patch(arrow3)
            continue
        ax1.add_patch(arrow)
    
    # Stats
    ax1.text(5, 0.5, '📊 Image Size: 128×128\n⚙️ Tasks: 1 (Hierarchical)\n🔧 Params: ~26M', 
             ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray'))
    
    # Multi-Task Model Components
    # Input
    ax2.text(5, 11, 'Input\n224×224×3', ha='center', va='center', fontsize=11, 
             bbox=input_style, fontweight='bold')
    
    # Backbone
    ax2.text(5, 9, 'ResNet50\n+ CBAM', ha='center', va='center', fontsize=11, 
             bbox=box_style, fontweight='bold')
    
    # Feature
    ax2.text(5, 7, 'Features\n2048-d', ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    # Outputs (4 tasks)
    y_positions_mt = [5, 3.8, 2.6, 1.4]
    outputs_mt = ['Main Category\n(2 classes)', 'Document Type\n(3 classes)', 
                  'Text Direction\n(2 classes)', '🔄 Rotation\n(4 classes)']
    colors_mt = [output_style, output_style, output_style, 
                 dict(boxstyle='round,pad=0.5', facecolor='#ffb3e6', edgecolor='purple', linewidth=2)]
    
    for y, output, style in zip(y_positions_mt, outputs_mt, colors_mt):
        ax2.text(5, y, output, ha='center', va='center', fontsize=9.5, 
                bbox=style, fontweight='bold')
    
    # Arrows
    arrow1 = FancyArrowPatch((5, 10.5), (5, 9.5), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='black')
    arrow2 = FancyArrowPatch((5, 8.5), (5, 7.5), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='black')
    ax2.add_patch(arrow1)
    ax2.add_patch(arrow2)
    
    # Branch arrows
    for y in y_positions_mt:
        arrow = FancyArrowPatch((5, 6.5), (5, y+0.5), arrowstyle='->', 
                               mutation_scale=15, linewidth=1.5, color='navy')
        ax2.add_patch(arrow)
    
    # Stats
    ax2.text(5, 0.2, '📊 Image Size: 224×224\n⚙️ Tasks: 2 (Hierarchical + Rotation)\n🔧 Params: ~26M', 
             ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray'))
    
    plt.tight_layout()
    return fig

def create_features_comparison():
    """Create bar chart comparing features"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    features = ['Image\nSize', 'Tasks', 'Rotation\nDetection', 'Auto Image\nCorrection', 
                'Hierarchical\nClassification']
    dhc_values = [128, 1, 0, 0, 1]
    multitask_values = [224, 2, 1, 1, 1]
    
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dhc_values, width, label='DHC Model', 
                   color='skyblue', edgecolor='navy', linewidth=1.5)
    bars2 = ax.bar(x + width/2, multitask_values, width, label='Multi-Task Model', 
                   color='lightcoral', edgecolor='red', linewidth=1.5)
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('🏛️ Feature Comparison: DHC vs Multi-Task Model', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                label = f'{int(height)}' if height != 1 or bar in bars1[1:3] else ('✓' if height == 1 else '✗')
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 Creating Model Comparison Visualizations...")
    
    # Create architecture comparison
    fig1 = create_comparison_diagram()
    fig1.savefig('model_architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_architecture_comparison.png")
    
    # Create features comparison
    fig2 = create_features_comparison()
    fig2.savefig('model_features_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_features_comparison.png")
    
    print("\n🎉 Visualizations created successfully!")
    print("📁 Check the following files:")
    print("   - model_architecture_comparison.png")
    print("   - model_features_comparison.png")
    
    plt.show()
