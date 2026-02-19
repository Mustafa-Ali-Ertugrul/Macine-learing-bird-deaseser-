#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential Training of All Models with Results Compilation
Trains: ViT-B/16, ResNeXt-50, ResNeSt-50d, ConvNeXt, CvT
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
import pandas as pd

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Model configurations
MODELS = [
    {
        'name': 'ViT-B/16',
        'script': 'train_vit_b16.py',
        'output_dir': 'vit_poultry_results',
        'log_file': 'vit_training_log.txt'
    },
    {
        'name': 'ResNeXt-50',
        'script': 'train_resnext.py',
        'output_dir': 'resnext_poultry_results',
        'log_file': 'resnext_training_log.txt'
    },
    {
        'name': 'ResNeSt-50d',
        'script': 'train_resnest.py',
        'output_dir': 'resnest_poultry_results',
        'log_file': 'resnest_training_log.txt'
    },
    {
        'name': 'ConvNeXt',
        'script': 'train_convnext.py',
        'output_dir': 'convnext_poultry_results',
        'log_file': 'convnext_training_log.txt'
    },
    {
        'name': 'CvT',
        'script': 'train_cvt.py',
        'output_dir': 'cvt_poultry_results',
        'log_file': 'cvt_training_log.txt'
    }
]

def train_model(model_config):
    """Train a single model and capture results"""
    print("\n" + "=" * 70)
    print(f"🚀 STARTING: {model_config['name']}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run training script
    try:
        with open(model_config['log_file'], 'w', encoding='utf-8') as log:
            result = subprocess.run(
                [sys.executable, model_config['script']],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8'
            )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_config['name']} completed successfully!")
            print(f"⏱️  Training time: {elapsed_time/60:.2f} minutes")
            return {
                'model': model_config['name'],
                'status': 'SUCCESS',
                'time_minutes': round(elapsed_time/60, 2),
                'output_dir': model_config['output_dir']
            }
        else:
            print(f"❌ {model_config['name']} failed with return code {result.returncode}")
            return {
                'model': model_config['name'],
                'status': 'FAILED',
                'time_minutes': round(elapsed_time/60, 2),
                'error': f"Exit code {result.returncode}"
            }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Exception during {model_config['name']}: {str(e)}")
        return {
            'model': model_config['name'],
            'status': 'ERROR',
            'time_minutes': round(elapsed_time/60, 2),
            'error': str(e)
        }

def extract_results(model_config):
    """Extract training results from output directory"""
    results = {
        'model': model_config['name'],
        'test_accuracy': None,
        'best_val_accuracy': None,
        'final_val_loss': None
    }
    
    # Try to read training log for metrics
    try:
        with open(model_config['log_file'], 'r', encoding='utf-8') as f:
            log_content = f.read()
            
            # Extract test accuracy (simplified - adjust based on actual output)
            if 'Test Accuracy:' in log_content:
                for line in log_content.split('\n'):
                    if 'Test Accuracy:' in line:
                        # Extract percentage
                        import re
                        match = re.search(r'(\d+\.\d+)%', line)
                        if match:
                            results['test_accuracy'] = float(match.group(1))
    except Exception as e:
        print(f"⚠️  Could not extract results for {model_config['name']}: {e}")
    
    return results

def main():
    print("\n" + "=" * 70)
    print("🎯 MULTI-MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 Number of models: {len(MODELS)}")
    print("\nModels to train:")
    for i, model in enumerate(MODELS, 1):
        print(f"   {i}. {model['name']}")
    print("=" * 70)
    
    overall_start = time.time()
    training_results = []
    
    # Train each model
    for model_config in MODELS:
        result = train_model(model_config)
        training_results.append(result)
        
        # Small delay between models
        time.sleep(2)
    
    overall_elapsed = time.time() - overall_start
    
    # Compile results
    print("\n" + "=" * 70)
    print("📊 TRAINING RESULTS SUMMARY")
    print("=" * 70)
    
    for result in training_results:
        status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_emoji} {result['model']}: {result['status']} ({result['time_minutes']} min)")
        if 'error' in result:
            print(f"   Error: {result['error']}")
    
    print(f"\n⏱️  Total pipeline time: {overall_elapsed/60:.2f} minutes")
    
    # Extract detailed results for successful models
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE METRICS")
    print("=" * 70)
    
    detailed_results = []
    for model_config in MODELS:
        if any(r['model'] == model_config['name'] and r['status'] == 'SUCCESS' for r in training_results):
            metrics = extract_results(model_config)
            detailed_results.append(metrics)
            
            if metrics['test_accuracy']:
                print(f"{model_config['name']}: {metrics['test_accuracy']:.2f}% accuracy")
            else:
                print(f"{model_config['name']}: Results pending analysis")
    
    # Save results to JSON
    results_file = 'all_models_training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'training_summary': training_results,
            'performance_metrics': detailed_results,
            'total_time_minutes': round(overall_elapsed/60, 2),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Summary
    successful = sum(1 for r in training_results if r['status'] == 'SUCCESS')
    print("\n" + "=" * 70)
    print(f"🏁 PIPELINE COMPLETE: {successful}/{len(MODELS)} models trained successfully")
    print("=" * 70)

if __name__ == '__main__':
    main()
