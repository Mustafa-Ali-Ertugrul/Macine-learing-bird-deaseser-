#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete All Model Trainings and Generate Final Report
Monitors running trainings and starts remaining models
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

MODELS = [
    {'name': 'ViT-B/16', 'script': 'train_vit_b16.py', 'output_dir': 'vit_poultry_results'},
    {'name': 'ResNeXt-50', 'script': 'train_resnext.py', 'output_dir': 'resnext_poultry_results'},
    {'name': 'ResNeSt-50d', 'script': 'train_resnest.py', 'output_dir': 'resnest_poultry_results'},
    {'name': 'ConvNeXt-Tiny', 'script': 'train_convnext.py', 'output_dir': 'convnext_poultry_results'},
    {'name': 'CvT-13', 'script': 'train_cvt.py', 'output_dir': 'cvt_poultry_results'}
]

def check_completion(output_dir):
    """Check if model training is complete"""
    if not os.path.exists(output_dir):
        return False
    
    # Check for final model and confusion matrix
    final_model = os.path.join(output_dir, 'final_model')
    best_model = os.path.join(output_dir, 'best_model.pth')
    confusion_matrix = os.path.join(output_dir, 'confusion_matrix.png')
    
    model_exists = os.path.exists(final_model) or os.path.exists(best_model)
    cm_exists = os.path.exists(confusion_matrix)
    
    return model_exists and cm_exists

def get_status():
    """Get current status of all models"""
    status = {}
    for model in MODELS:
        if check_completion(model['output_dir']):
            status[model['name']] = 'COMPLETE'
        elif os.path.exists(model['output_dir']):
            # Check for checkpoints
            try:
                files = os.listdir(model['output_dir'])
                if any('checkpoint' in f for f in files):
                    status[model['name']] = 'IN_PROGRESS'
                else:
                    status[model['name']] = 'STARTED'
            except:
                status[model['name']] = 'STARTED'
        else:
            status[model['name']] = 'PENDING'
    return status

def train_remaining_models():
    """Train any remaining models that haven't been started"""
    print("\n" + "=" * 80)
    print("🚀 CHECKING FOR REMAINING MODELS TO TRAIN")
    print("=" * 80)
    
    status = get_status()
    
    for model in MODELS:
        model_status = status[model['name']]
        
        if model_status == 'PENDING':
            print(f"\n⏯️  Starting training for {model['name']}...")
            print("-" * 80)
            
            try:
                # Run training script
                result = subprocess.run(
                    [sys.executable, model['script']],
                    capture_output=False,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"✅ {model['name']} training completed successfully!")
                else:
                    print(f"⚠️  {model['name']} training finished with return code {result.returncode}")
            
            except Exception as e:
                print(f"❌ Error training {model['name']}: {e}")
            
            # Small delay between trainings
            time.sleep(5)
        
        elif model_status == 'IN_PROGRESS':
            print(f"🔄 {model['name']}: Currently in progress (running in another terminal)")
        
        elif model_status == 'COMPLETE':
            print(f"✅ {model['name']}: Already completed")
        
        else:
            print(f"⏸️  {model['name']}: Status - {model_status}")

def wait_for_completion(check_interval=60):
    """Wait for all trainings to complete"""
    print("\n" + "=" * 80)
    print("⏰ WAITING FOR ALL TRAININGS TO COMPLETE")
    print("=" * 80)
    print(f"Checking every {check_interval} seconds... (Press Ctrl+C to skip waiting)")
    
    try:
        while True:
            status = get_status()
            
            all_complete = all(s == 'COMPLETE' for s in status.values())
            
            if all_complete:
                print("\n✅ All trainings completed!")
                return True
            
            # Display current status
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status:")
            for model_name, model_status in status.items():
                emoji = "✅" if model_status == "COMPLETE" else "🔄" if model_status == "IN_PROGRESS" else "⏸️"
                print(f"  {emoji} {model_name}: {model_status}")
            
            completed = sum(1 for s in status.values() if s == 'COMPLETE')
            print(f"\nProgress: {completed}/{len(MODELS)} models completed")
            
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\n⏭️  Skipping wait - generating report with available data")
        return False

def generate_final_report():
    """Generate comprehensive final report"""
    print("\n" + "=" * 80)
    print("📊 GENERATING FINAL COMPREHENSIVE REPORT")
    print("=" * 80)
    
    # Run results extraction
    print("\n1️⃣  Extracting results from all models...")
    try:
        subprocess.run([sys.executable, 'extract_checkpoint_metrics.py'], check=True)
        print("   ✅ Results extracted")
    except Exception as e:
        print(f"   ⚠️  Error extracting results: {e}")
    
    # Generate HTML report
    print("\n2️⃣  Generating HTML report and visualizations...")
    try:
        subprocess.run([sys.executable, 'create_final_report.py'], check=True)
        print("   ✅ Report generated")
    except Exception as e:
        print(f"   ⚠️  Error generating report: {e}")
    
    print("\n" + "=" * 80)
    print("✅ FINAL REPORT GENERATION COMPLETE")
    print("=" * 80)
    print("\n📁 Output files:")
    print("   • training_results_summary.json - Raw results data")
    print("   • model_comparison.png - Performance comparison chart")
    print("   • training_report.html - Interactive HTML report")
    print("\n💡 Open 'training_report.html' in your browser to view the full report")
    print("=" * 80)

def main():
    print("\n" + "=" * 80)
    print("🎯 COMPLETE ALL MODEL TRAININGS - AUTOMATED PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check initial status
    print("\n📋 Initial Status Check:")
    status = get_status()
    for model_name, model_status in status.items():
        emoji = "✅" if model_status == "COMPLETE" else "🔄" if model_status == "IN_PROGRESS" else "⏸️"
        print(f"  {emoji} {model_name}: {model_status}")
    
    # Step 2: Train remaining models
    train_remaining_models()
    
    # Step 3: Wait for all to complete (optional)
    print("\n" + "=" * 80)
    choice = input("⏰ Wait for all trainings to complete before generating report? (y/n): ").strip().lower()
    
    if choice == 'y':
        wait_for_completion(check_interval=60)
    else:
        print("⏭️  Proceeding to report generation with currently available data")
    
    # Step 4: Generate final report
    generate_final_report()
    
    print(f"\n🏁 Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
