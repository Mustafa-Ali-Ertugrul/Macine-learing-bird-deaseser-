#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Training Monitor - Display progress and results
"""

import os
import sys
import time
import json
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

MODELS = [
    {'name': 'ViT-B/16', 'output_dir': 'vit_poultry_results'},
    {'name': 'ResNeXt-50', 'output_dir': 'resnext_poultry_results'},
    {'name': 'ResNeSt-50d', 'output_dir': 'resnest_poultry_results'},
    {'name': 'ConvNeXt-Tiny', 'output_dir': 'convnext_poultry_results'},
    {'name': 'CvT-13', 'output_dir': 'cvt_poultry_results'}
]

def check_model_status(model_config):
    """Check if model training has completed"""
    output_dir = model_config['output_dir']
    
    if not os.path.exists(output_dir):
        return {'status': 'NOT_STARTED', 'progress': 0}
    
    final_model_path = os.path.join(output_dir, 'final_model')
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    
    if os.path.exists(final_model_path) and os.path.exists(confusion_matrix_path):
        return {'status': 'COMPLETED', 'progress': 100}
    elif os.path.exists(output_dir):
        # Check for checkpoint files
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint')]
        if checkpoints:
            return {'status': 'IN_PROGRESS', 'progress': 50, 'checkpoints': len(checkpoints)}
        else:
            return {'status': 'STARTED', 'progress': 10}
    
    return {'status': 'NOT_STARTED', 'progress': 0}

def display_status():
    """Display current status of all models"""
    print("\n" + "=" * 70)
    print(f"🔍 TRAINING STATUS CHECK - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    results_summary = []
    
    for model in MODELS:
        status_info = check_model_status(model)
        
        # Status emoji
        if status_info['status'] == 'COMPLETED':
            emoji = "✅"
        elif status_info['status'] in ['IN_PROGRESS', 'STARTED']:
            emoji = "🔄"
        else:
            emoji = "⏸️"
        
        status_line = f"{emoji} {model['name']:<20} {status_info['status']:<15}"
        
        if status_info['status'] == 'IN_PROGRESS':
            status_line += f" ({status_info.get('checkpoints', 0)} checkpoints)"
        
        print(status_line)
        results_summary.append({
            'model': model['name'],
            'status': status_info['status'],
            'progress': status_info['progress']
        })
    
    # Overall progress
    total_progress = sum(r['progress'] for r in results_summary) / len(results_summary)
    print("\n" + "=" * 70)
    print(f"📊 Overall Progress: {total_progress:.1f}%")
    print("=" * 70)
    
    return results_summary

def extract_accuracy_from_results(model_config):
    """Try to extract test accuracy from model results"""
    output_dir = model_config['output_dir']
    
    # Try to find trainer_state.json
    trainer_state_path = os.path.join(output_dir, 'final_model', 'trainer_state.json')
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
                if 'log_history' in state:
                    # Find last eval accuracy
                    for entry in reversed(state['log_history']):
                        if 'eval_accuracy' in entry:
                            return entry['eval_accuracy']
        except:
            pass
    
    return None

def display_results():
    """Display final results if available"""
    print("\n" + "=" * 70)
    print("📈 FINAL RESULTS (where available)")
    print("=" * 70)
    
    for model in MODELS:
        status_info = check_model_status(model)
        
        if status_info['status'] == 'COMPLETED':
            accuracy = extract_accuracy_from_results(model)
            
            if accuracy:
                print(f"✅ {model['name']:<20} Test Accuracy: {accuracy*100:.2f}%")
            else:
                print(f"✅ {model['name']:<20} Training completed (metrics in {model['output_dir']})")
        else:
            print(f"⏸️  {model['name']:<20} {status_info['status']}")
    
    print("=" * 70)

if __name__ == '__main__':
    # Monitor mode
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        print("🔍 Monitoring mode - Press Ctrl+C to stop")
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                display_status()
                time.sleep(30)  # Update every 30 seconds
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
    else:
        # Single check
        display_status()
        print()
        display_results()
