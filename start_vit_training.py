#!/usr/bin/env python3
"""
Quick Start Script for ViT-B/16 Training
This script checks dependencies and starts the training process
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'transformers',
        'timm',
        'PIL',
        'sklearn',
        'tqdm'
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def check_data_directory():
    """Check if resized dataset exists"""
    data_dir = 'poultry_dataset_512x512/poultry_microscopy'
    
    print(f"\n📁 Checking dataset directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"   ❌ Directory not found!")
        print(f"\n⚠️  Please run resize_images_512.py first to create 512x512 resized images")
        return False
    
    # Count subdirectories (disease classes)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(classes) == 0:
        print(f"   ❌ No disease class folders found!")
        return False
    
    print(f"   ✅ Found {len(classes)} disease categories")
    
    # Count total images
    total_images = 0
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
        total_images += len(images)
    
    print(f"   ✅ Found {total_images:,} total images")
    
    return True

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n🎮 GPU Available: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            
            # Recommend batch size based on memory
            if gpu_memory >= 12:
                recommended_batch = 32
            elif gpu_memory >= 8:
                recommended_batch = 16
            elif gpu_memory >= 6:
                recommended_batch = 8
            else:
                recommended_batch = 4
            
            print(f"   Recommended batch size: {recommended_batch}")
            return True
        else:
            print("\n⚠️  No GPU detected - Training will be slow on CPU")
            print("   Consider using Google Colab with free GPU")
            return False
    except:
        return False

def main():
    """Main function"""
    print("=" * 70)
    print("ViT-B/16 POULTRY DISEASE CLASSIFICATION - PRE-FLIGHT CHECK")
    print("=" * 70)
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print("\n" + "=" * 70)
        print("⚠️  MISSING PACKAGES DETECTED")
        print("=" * 70)
        print("\nPlease install missing packages:")
        print("\n   pip install -r requirements.txt")
        print("\nOr install individually:")
        for package in missing_packages:
            if package == 'PIL':
                print(f"   pip install Pillow")
            elif package == 'sklearn':
                print(f"   pip install scikit-learn")
            else:
                print(f"   pip install {package}")
        
        print("\n" + "=" * 70)
        return
    
    # Check dataset
    if not check_data_directory():
        print("\n" + "=" * 70)
        print("⚠️  DATASET NOT READY")
        print("=" * 70)
        print("\nPlease run the resize script first:")
        print("\n   python resize_images_512.py")
        print("\nThis will create standardized 512x512 images for training.")
        print("=" * 70)
        return
    
    # Check GPU
    has_gpu = check_gpu()
    
    # All checks passed
    print("\n" + "=" * 70)
    print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
    print("=" * 70)
    
    print("\n🚀 To start training, run:")
    print("\n   python train_vit_b16.py")
    
    if not has_gpu:
        print("\n⚠️  Training on CPU will be very slow (1-2 hours per epoch)")
        print("   Consider using Google Colab for free GPU access")
    else:
        print("\n⏱️  Estimated training time with GPU:")
        print("   - ~8-12 minutes per epoch")
        print("   - Total: ~2-3 hours for 15 epochs")
    
    print("\n📊 After training, use for prediction:")
    print("\n   python predict_single.py <image_path>")
    
    print("\n" + "=" * 70)
    
    # Ask if user wants to start training now
    try:
        response = input("\n❓ Start training now? (y/n): ").strip().lower()
        if response == 'y':
            print("\n🚀 Starting training...\n")
            subprocess.run([sys.executable, 'train_vit_b16.py'])
        else:
            print("\n👍 Run 'python train_vit_b16.py' when ready!")
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled. Run 'python train_vit_b16.py' when ready!")

if __name__ == '__main__':
    main()
