"""
Comprehensive Dataset Validation Script
Checks for data leakage, overfitting, class imbalance, and other issues
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DATASET_DIRS = [
    'Macine learing (bird deaseser)/final_dataset_10_classes',
    'data',
    'organized_poultry_dataset',
    'poultry_microscopy'
]

class DatasetValidator:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
        
    def log_issue(self, severity, message):
        """Log an issue found during validation"""
        if severity == "ERROR":
            self.issues.append(f"🚨 {message}")
        else:
            self.warnings.append(f"⚠️  {message}")
        print(f"{'🚨' if severity == 'ERROR' else '⚠️'} {message}")
    
    def check_dataset_exists(self):
        """Check which dataset directories exist"""
        print("\n" + "=" * 80)
        print("1️⃣  CHECKING DATASET DIRECTORIES")
        print("=" * 80)
        
        found_dirs = []
        for dir_path in DATASET_DIRS:
            if Path(dir_path).exists():
                found_dirs.append(dir_path)
                print(f"✅ Found: {dir_path}")
            else:
                print(f"❌ Not found: {dir_path}")
        
        if not found_dirs:
            self.log_issue("ERROR", "No dataset directories found!")
            return None
        
        # Use the first found directory (prioritized list)
        return found_dirs[0]
    
    def analyze_class_distribution(self, data_dir):
        """Analyze class balance and distribution"""
        print("\n" + "=" * 80)
        print("2️⃣  ANALYZING CLASS DISTRIBUTION")
        print("=" * 80)
        
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if not classes:
            self.log_issue("ERROR", "No class directories found in dataset!")
            return None
        
        class_counts = {}
        for cls in classes:
            cls_path = os.path.join(data_dir, cls)
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            class_counts[cls] = len(images)
        
        # Statistics
        counts = list(class_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = min(counts)
        max_count = max(counts)
        
        print(f"\n📊 Class Statistics:")
        print(f"   Total Classes: {len(classes)}")
        print(f"   Mean samples per class: {mean_count:.1f}")
        print(f"   Std deviation: {std_count:.1f}")
        print(f"   Min samples: {min_count}")
        print(f"   Max samples: {max_count}")
        print(f"   Imbalance ratio: {max_count/min_count:.2f}x")
        
        print(f"\n📈 Per-Class Breakdown:")
        print(f"   {'Class':<30} {'Count':>8}  {'% of Total':>10}")
        print(f"   {'-'*30} {'-'*8}  {'-'*10}")
        
        total_images = sum(counts)
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_images) * 100
            print(f"   {cls:<30} {count:>8,}  {pct:>9.1f}%")
        
        # Check for severe imbalance
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 10:
            self.log_issue("ERROR", f"Severe class imbalance detected! Ratio: {imbalance_ratio:.1f}x")
        elif imbalance_ratio > 5:
            self.log_issue("WARNING", f"Moderate class imbalance detected. Ratio: {imbalance_ratio:.1f}x")
        
        # Check for very small classes
        for cls, count in class_counts.items():
            if count < 50:
                self.log_issue("ERROR", f"Class '{cls}' has only {count} samples - too small for reliable training!")
            elif count < 200:
                self.log_issue("WARNING", f"Class '{cls}' has only {count} samples - may cause overfitting")
        
        self.stats['class_counts'] = class_counts
        self.stats['total_images'] = total_images
        
        return class_counts
    
    def check_data_leakage(self, data_dir, sample_ratio=1.0):
        """Check for duplicate images between train/test splits using perceptual hashing"""
        print("\n" + "=" * 80)
        print("3️⃣  CHECKING FOR DATA LEAKAGE")
        print("=" * 80)
        
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        total_duplicates = 0
        duplicate_details = []
        
        for cls in classes:
            print(f"\n🔍 Analyzing class: {cls}")
            cls_path = os.path.join(data_dir, cls)
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            
            # Sample if too many images
            if sample_ratio < 1.0:
                sample_size = int(len(images) * sample_ratio)
                images = np.random.choice(images, sample_size, replace=False)
            
            # Simulate train/test split with same random seed as training
            np.random.seed(42)
            np.random.shuffle(images)
            split_idx = int(len(images) * 0.8)
            train_imgs = images[:split_idx]
            test_imgs = images[split_idx:]
            
            print(f"   Train: {len(train_imgs)}, Test: {len(test_imgs)}")
            
            # Calculate perceptual hashes for training set
            train_hashes = {}
            print(f"   Hashing training images...")
            for img_path in tqdm(train_imgs, desc="Train", leave=False):
                try:
                    img = Image.open(img_path)
                    h = imagehash.phash(img)
                    if h in train_hashes:
                        # Duplicate within training set
                        self.log_issue("WARNING", f"Duplicate in training set: {os.path.basename(img_path)}")
                    train_hashes[h] = img_path
                except Exception as e:
                    continue
            
            # Check test set against training set
            leaks = []
            print(f"   Checking test images for leakage...")
            for img_path in tqdm(test_imgs, desc="Test", leave=False):
                try:
                    img = Image.open(img_path)
                    h = imagehash.phash(img)
                    
                    # Check exact match
                    if h in train_hashes:
                        leaks.append({
                            'test_image': os.path.basename(img_path),
                            'train_image': os.path.basename(train_hashes[h]),
                            'class': cls,
                            'distance': 0
                        })
                    else:
                        # Check for near-duplicates (hamming distance <= 3)
                        for train_hash, train_path in train_hashes.items():
                            dist = h - train_hash
                            if dist <= 3:
                                leaks.append({
                                    'test_image': os.path.basename(img_path),
                                    'train_image': os.path.basename(train_path),
                                    'class': cls,
                                    'distance': dist
                                })
                                break
                except Exception as e:
                    continue
            
            if leaks:
                print(f"   🚨 Found {len(leaks)} potential leaks in {cls}!")
                total_duplicates += len(leaks)
                duplicate_details.extend(leaks)
                
                # Show first 3 examples
                for i, leak in enumerate(leaks[:3]):
                    print(f"      Leak {i+1}: Test '{leak['test_image']}' ≈ Train '{leak['train_image']}' (distance: {leak['distance']})")
                    
                leak_ratio = len(leaks) / len(test_imgs) * 100
                if leak_ratio > 10:
                    self.log_issue("ERROR", f"Class '{cls}': {leak_ratio:.1f}% of test set leaked from training!")
                elif leak_ratio > 5:
                    self.log_issue("WARNING", f"Class '{cls}': {leak_ratio:.1f}% of test set may be leaked")
            else:
                print(f"   ✅ No leakage detected in {cls}")
        
        print(f"\n📊 Leakage Summary:")
        print(f"   Total potential leaks: {total_duplicates}")
        
        if total_duplicates > 0:
            self.log_issue("ERROR", f"CRITICAL: {total_duplicates} duplicate images found between train/test splits!")
            self.log_issue("ERROR", "This causes data leakage and inflates validation metrics!")
            
            # Save leak details
            leak_df = pd.DataFrame(duplicate_details)
            leak_df.to_csv('data_leakage_report.csv', index=False)
            print(f"\n   📁 Saved detailed leak report: data_leakage_report.csv")
        
        self.stats['total_leaks'] = total_duplicates
        self.stats['leak_details'] = duplicate_details
        
        return total_duplicates
    
    def check_image_quality(self, data_dir, sample_size=100):
        """Check for corrupted or low-quality images"""
        print("\n" + "=" * 80)
        print("4️⃣  CHECKING IMAGE QUALITY")
        print("=" * 80)
        
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        corrupted_images = []
        small_images = []
        
        for cls in classes:
            cls_path = os.path.join(data_dir, cls)
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            
            # Sample random images
            sample = np.random.choice(images, min(sample_size, len(images)), replace=False)
            
            for img_path in sample:
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    
                    # Check for very small images
                    if width < 100 or height < 100:
                        small_images.append({
                            'path': img_path,
                            'size': f"{width}x{height}",
                            'class': cls
                        })
                except Exception as e:
                    corrupted_images.append({
                        'path': img_path,
                        'error': str(e),
                        'class': cls
                    })
        
        print(f"\n📊 Quality Check Results:")
        print(f"   Corrupted images: {len(corrupted_images)}")
        print(f"   Small images (<100px): {len(small_images)}")
        
        if corrupted_images:
            self.log_issue("ERROR", f"Found {len(corrupted_images)} corrupted images!")
            for corrupt in corrupted_images[:5]:
                print(f"      {corrupt['path']}: {corrupt['error']}")
        
        if small_images:
            self.log_issue("WARNING", f"Found {len(small_images)} very small images")
        
        return len(corrupted_images), len(small_images)
    
    def analyze_split_strategy(self, data_dir):
        """Analyze the train/validation/test split strategy"""
        print("\n" + "=" * 80)
        print("5️⃣  ANALYZING SPLIT STRATEGY")
        print("=" * 80)
        
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Simulate the split used in training scripts
        all_images = []
        all_labels = []
        
        for cls in classes:
            cls_path = os.path.join(data_dir, cls)
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            all_images.extend(images)
            all_labels.extend([cls] * len(images))
        
        # Replicate the split from training scripts (80/10/10 or 70/15/15)
        print(f"\n📊 Simulating split with random_state=42:")
        
        # Split 1: train+val vs test (80/20)
        train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(
            all_images, all_labels,
            test_size=0.2,
            stratify=all_labels,
            random_state=42
        )
        
        # Split 2: train vs val (90/10 of remaining)
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            train_val_imgs, train_val_labels,
            test_size=0.1,
            stratify=train_val_labels,
            random_state=42
        )
        
        print(f"   Training set: {len(train_imgs)} ({len(train_imgs)/len(all_images)*100:.1f}%)")
        print(f"   Validation set: {len(val_imgs)} ({len(val_imgs)/len(all_images)*100:.1f}%)")
        print(f"   Test set: {len(test_imgs)} ({len(test_imgs)/len(all_images)*100:.1f}%)")
        
        # Check class distribution in each split
        print(f"\n   Class distribution per split:")
        train_class_counts = Counter(train_labels)
        val_class_counts = Counter(val_labels)
        test_class_counts = Counter(test_labels)
        
        print(f"   {'Class':<30} {'Train':>8} {'Val':>8} {'Test':>8}")
        print(f"   {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        
        for cls in classes:
            print(f"   {cls:<30} {train_class_counts[cls]:>8} {val_class_counts[cls]:>8} {test_class_counts[cls]:>8}")
        
        # Check if validation set is too small
        if len(val_imgs) < 100:
            self.log_issue("WARNING", f"Validation set is very small ({len(val_imgs)} images) - may cause unstable metrics")
        
        return {
            'train_size': len(train_imgs),
            'val_size': len(val_imgs),
            'test_size': len(test_imgs)
        }
    
    def generate_report(self):
        """Generate final validation report"""
        print("\n" + "=" * 80)
        print("📋 DATASET VALIDATION REPORT")
        print("=" * 80)
        
        print(f"\n🔴 CRITICAL ISSUES ({len(self.issues)}):")
        if self.issues:
            for issue in self.issues:
                print(f"   {issue}")
        else:
            print("   ✅ No critical issues found")
        
        print(f"\n🟡 WARNINGS ({len(self.warnings)}):")
        if self.warnings:
            for warning in self.warnings:
                print(f"   {warning}")
        else:
            print("   ✅ No warnings")
        
        # Overall assessment
        print("\n" + "=" * 80)
        print("🎯 OVERALL ASSESSMENT")
        print("=" * 80)
        
        if len(self.issues) > 0:
            print("❌ Dataset has CRITICAL ISSUES that likely explain high validation scores:")
            print("   - Data leakage inflates validation/test metrics")
            print("   - Model memorizes duplicates instead of learning features")
            print("   - Real-world performance will be much lower")
            print("\n💡 RECOMMENDED ACTIONS:")
            print("   1. Remove duplicate images from dataset")
            print("   2. Re-split dataset ensuring no leakage")
            print("   3. Retrain models with clean dataset")
            print("   4. Verify new validation scores are realistic")
        elif len(self.warnings) > 0:
            print("⚠️  Dataset has minor issues but should be usable")
            print("   - Consider addressing warnings for better model performance")
        else:
            print("✅ Dataset appears healthy!")
            print("   - If validation scores are still unusually high, check:")
            print("     * Model architecture for overfitting")
            print("     * Training hyperparameters")
            print("     * Task difficulty (may actually be easy)")
        
        # Save report
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        report_data = {
            'issues': self.issues,
            'warnings': self.warnings,
            'stats': convert_to_serializable(self.stats)
        }
        
        with open('dataset_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Detailed report saved: dataset_validation_report.json")
        print("=" * 80)

def main():
    print("\n" + "=" * 80)
    print("🔬 COMPREHENSIVE DATASET VALIDATION")
    print("   Checking for data leakage, class imbalance, and other issues")
    print("=" * 80)
    
    validator = DatasetValidator()
    
    # 1. Find dataset
    data_dir = validator.check_dataset_exists()
    if not data_dir:
        print("\n❌ Cannot proceed without dataset directory")
        return
    
    # 2. Analyze class distribution
    class_counts = validator.analyze_class_distribution(data_dir)
    if not class_counts:
        print("\n❌ Cannot proceed without class data")
        return
    
    # 3. Check for data leakage (most critical for high validation scores)
    print("\n⚠️  This may take several minutes depending on dataset size...")
    validator.check_data_leakage(data_dir, sample_ratio=1.0)  # Check all images
    
    # 4. Check image quality
    validator.check_image_quality(data_dir, sample_size=100)
    
    # 5. Analyze split strategy
    validator.analyze_split_strategy(data_dir)
    
    # 6. Generate final report
    validator.generate_report()

if __name__ == "__main__":
    main()
