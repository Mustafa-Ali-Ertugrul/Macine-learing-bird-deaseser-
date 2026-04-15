#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Remaining Models (Skipping ViT)
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

MODELS = [
    {
        'name': 'ResNeXt-50',
        'script': 'train_resnext.py',
        'log_file': 'resnext_training_log.txt'
    },
    {
        'name': 'ResNeSt-50d',
        'script': 'train_resnest.py',
        'log_file': 'resnest_training_log.txt'
    },
    {
        'name': 'ConvNeXt',
        'script': 'train_convnext.py',
        'log_file': 'convnext_training_log.txt'
    },
    {
        'name': 'CvT',
        'script': 'train_cvt.py',
        'log_file': 'cvt_training_log.txt'
    }
]

def train_model(model_config):
    print("\n" + "=" * 70)
    print(f"🚀 STARTING: {model_config['name']}")
    print("=" * 70)
    
    start_time = time.time()
    
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
            return True
        else:
            print(f"❌ {model_config['name']} failed with return code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"❌ Exception during {model_config['name']}: {str(e)}")
        return False

def main():
    print("🎯 RESTARTING FAILED TRAININGS")
    
    for model in MODELS:
        train_model(model)
        time.sleep(2)

if __name__ == '__main__':
    main()
