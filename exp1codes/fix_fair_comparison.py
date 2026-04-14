"""
Script to fix the fair comparison issue in table1_experiment.ipynb
This adds ResNet-50 Enhanced version and parameter counting for fair comparison
"""

import json
import sys

def fix_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find cells and modify
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        
        # Cell 3: Add count_parameters function
        if '# Evaluation function to compute all metrics' in source and 'def evaluate_model' in source:
            new_source = source.replace(
                '# Evaluation function to compute all metrics',
                '# Function to count model parameters\n'
                'def count_parameters(model):\n'
                '    """Count the number of trainable parameters in a model"""\n'
                '    return sum(p.numel() for p in model.parameters() if p.requires_grad)\n\n'
                '# Evaluation function to compute all metrics'
            )
            cell['source'] = new_source.split('\n')
            print(f"✓ Updated cell {i}: Added count_parameters function")
        
        # Cell 8: Replace ResNet-50 with Standard + Enhanced versions
        if '# Load ResNet-50 and modify for CIFAR-100' in source:
            new_source = '''# ===== ResNet-50 Standard (Baseline) =====
# Standard ResNet-50 with single-layer classifier head
resnet50_standard = resnet50(pretrained=False)
resnet50_standard.fc = nn.Linear(2048, 100)  # Single layer: 2048 -> 100

# Training hyperparameters
resnet_epochs = 2 if QUICK_TEST else 100
resnet_lr = 0.001
resnet_weight_decay = 0.0001

print("="*60)
print("Training ResNet-50 (Standard - Single Layer Head)")
print("="*60)
resnet_std_params = count_parameters(resnet50_standard)
print(f"Parameters: {resnet_std_params:,}")

resnet_std_metrics = train_model(
    resnet50_standard, train_loader, test_loader,
    resnet_epochs, resnet_lr, resnet_weight_decay, device, 'ResNet-50 (Standard)'
)

print("\\nResNet-50 (Standard) Final Results:")
print(f"Accuracy: {resnet_std_metrics['accuracy']:.2f}%")
print(f"Precision: {resnet_std_metrics['precision']:.2f}%")
print(f"Recall: {resnet_std_metrics['recall']:.2f}%")
print(f"F1: {resnet_std_metrics['f1']:.2f}%")

# ===== ResNet-50 Enhanced (Fair Comparison) =====
# ResNet-50 with enhanced classifier head matching VDLF-Net's complexity
# This isolates the contribution of VDLF-Net's fusion mechanism
resnet50_enhanced = resnet50(pretrained=False)
# Use the same enhanced classifier head as VDLF-Net for fair comparison
resnet50_enhanced.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 100)
)

print("\\n" + "="*60)
print("Training ResNet-50 (Enhanced - Multi-Layer Head)")
print("="*60)
resnet_enh_params = count_parameters(resnet50_enhanced)
print(f"Parameters: {resnet_enh_params:,}")

resnet_enh_metrics = train_model(
    resnet50_enhanced, train_loader, test_loader,
    resnet_epochs, resnet_lr, resnet_weight_decay, device, 'ResNet-50 (Enhanced)'
)

print("\\nResNet-50 (Enhanced) Final Results:")
print(f"Accuracy: {resnet_enh_metrics['accuracy']:.2f}%")
print(f"Precision: {resnet_enh_metrics['precision']:.2f}%")
print(f"Recall: {resnet_enh_metrics['recall']:.2f}%")
print(f"F1: {resnet_enh_metrics['f1']:.2f}%")

# Use enhanced version for comparison (more fair)
resnet_metrics = resnet_enh_metrics
resnet50_model = resnet50_enhanced'''
            cell['source'] = new_source.split('\n')
            print(f"✓ Updated cell {i}: Replaced ResNet-50 with Standard + Enhanced versions")
        
        # Cell 13: Add parameter counting for VDLF-Net
        if '# Initialize VDLF-Net' in source and 'vdlfnet = VDLFNet' in source:
            new_source = source.replace(
                'print("Training VDLF-Net...")',
                'print("\\n" + "="*60)\n'
                'print("Training VDLF-Net")\n'
                'print("="*60)\n'
                'vdlf_params = count_parameters(vdlfnet)\n'
                'print(f"Parameters: {vdlf_params:,}")'
            )
            new_source = new_source.replace(
                'print(f"Hyperparameters: epochs={vdlf_epochs}, lr={vdlf_lr}, alpha={vdlf_alpha}")',
                'print(f"Hyperparameters: epochs={vdlf_epochs}, lr={vdlf_lr}, alpha={vdlf_alpha}")\n'
                '\n'
                '# Print parameter comparison\n'
                'print("\\n" + "="*60)\n'
                'print("Parameter Comparison:")\n'
                'print("="*60)\n'
                'print(f"ResNet-50 (Standard): {resnet_std_params:,} parameters")\n'
                'print(f"ResNet-50 (Enhanced): {resnet_enh_params:,} parameters")\n'
                'print(f"VDLF-Net:            {vdlf_params:,} parameters")\n'
                'print(f"\\nVDLF-Net vs ResNet-50 (Enhanced): +{vdlf_params - resnet_enh_params:,} parameters")\n'
                'print("(Additional parameters come from VAE encoder/decoder and fusion components)")\n'
                'print("="*60)'
            )
            cell['source'] = new_source.split('\n')
            print(f"✓ Updated cell {i}: Added parameter counting for VDLF-Net")
        
        # Cell 15: Update results table
        if '# Create Table 1 results' in source and 'import pandas as pd' in source:
            # This is complex, we'll provide instructions instead
            print(f"⚠ Cell {i}: Table results cell needs manual update (see FAIR_COMPARISON_FIX.md)")
    
    # Save modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Notebook updated: {notebook_path}")
    print("⚠ Please manually update the results table cell (Cell 15) as described in FAIR_COMPARISON_FIX.md")

if __name__ == '__main__':
    notebook_path = 'table1_experiment.ipynb'
    fix_notebook(notebook_path)
