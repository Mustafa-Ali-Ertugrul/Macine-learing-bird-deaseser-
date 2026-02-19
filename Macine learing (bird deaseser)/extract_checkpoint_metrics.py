#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract metrics from checkpoint files for in-progress trainings
"""

import os
import sys
import json
import glob

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def extract_from_checkpoints(output_dir):
    """Extract best metrics from checkpoint trainer_state.json files"""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
    
    best_metrics = {
        'best_val_accuracy': 0,
        'best_val_loss': float('inf'),
        'latest_epoch': 0,
        'checkpoints_found': len(checkpoint_dirs)
    }
    
    for ckpt_dir in checkpoint_dirs:
        trainer_state_file = os.path.join(ckpt_dir, 'trainer_state.json')
        
        if os.path.exists(trainer_state_file):
            try:
                with open(trainer_state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                    # Get latest epoch
                    if 'epoch' in state:
                        best_metrics['latest_epoch'] = state['epoch']
                    
                    # Parse log history
                    if 'log_history' in state:
                        for entry in state['log_history']:
                            if 'eval_accuracy' in entry:
                                acc = entry['eval_accuracy']
                                best_metrics['best_val_accuracy'] = max(
                                    best_metrics['best_val_accuracy'], acc
                                )
                            
                            if 'eval_loss' in entry:
                                loss = entry['eval_loss']
                                best_metrics['best_val_loss'] = min(
                                    best_metrics['best_val_loss'], loss
                                )
            
            except Exception as e:
                print(f"⚠️  Error reading {trainer_state_file}: {e}")
    
    return best_metrics

def main():
    print("\n" + "=" * 80)
    print("📊 EXTRACTING METRICS FROM CHECKPOINTS")
    print("=" * 80)
    
    models = {
        'ViT-B/16': 'vit_poultry_results',
        'ResNeXt-50': 'resnext_poultry_results',
        'ResNeSt-50d': 'resnest_poultry_results',
        'ConvNeXt-Tiny': 'convnext_poultry_results',
        'CvT-13': 'cvt_poultry_results'
    }
    
    all_results = {}
    
    for model_name, output_dir in models.items():
        print(f"\n🔍 {model_name}")
        print("-" * 80)
        
        if not os.path.exists(output_dir):
            print(f"  ⏸️  Not started (no output directory)")
            continue
        
        metrics = extract_from_checkpoints(output_dir)
        
        if metrics:
            all_results[model_name] = metrics
            
            print(f"  ✅ Found {metrics['checkpoints_found']} checkpoints")
            print(f"  📈 Best Val Accuracy: {metrics['best_val_accuracy']*100:.2f}%")
            print(f"  📉 Best Val Loss: {metrics['best_val_loss']:.4f}")
            print(f"  🔢 Latest Epoch: {metrics['latest_epoch']:.2f}")
        else:
            print(f"  ⚠️  No checkpoints found")
    
    # Save results
    output_file = 'checkpoint_metrics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"💾 Metrics saved to: {output_file}")
    print("=" * 80)
    
    # Summary
    print("\n📋 SUMMARY:")
    print("-" * 80)
    print(f"{'Model':<20} {'Val Accuracy':<15} {'Val Loss':<15} {'Epoch':<10}")
    print("-" * 80)
    
    for model_name, metrics in all_results.items():
        acc = f"{metrics['best_val_accuracy']*100:.2f}%"
        loss = f"{metrics['best_val_loss']:.4f}"
        epoch = f"{metrics['latest_epoch']:.2f}"
        print(f"{model_name:<20} {acc:<15} {loss:<15} {epoch:<10}")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
