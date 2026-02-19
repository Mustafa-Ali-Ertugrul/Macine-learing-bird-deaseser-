#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential Training of REMAINING Models
Trains: ConvNeXt, CvT
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Only the remaining models
MODELS = [
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

def main():
    print("\n" + "=" * 70)
    print("🎯 REMAINING MODELS TRAINING PIPELINE")
    print("=" * 70)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 Number of models: {len(MODELS)}")
    
    overall_start = time.time()
    for model in MODELS:
        train_model(model)
        time.sleep(2)
    
    overall_elapsed = time.time() - overall_start
    print(f"\n🏁 Finished in {overall_elapsed/60:.2f} minutes")

if __name__ == '__main__':
    main()
