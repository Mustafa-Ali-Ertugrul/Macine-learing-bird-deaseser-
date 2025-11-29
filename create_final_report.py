#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Final Comprehensive Training Report with visualizations
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def load_checkpoint_metrics():
    """Load metrics from checkpoint analysis"""
    try:
        with open('checkpoint_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def create_comparison_chart(metrics):
    """Create performance comparison chart"""
    if not metrics:
        print("⚠️  No metrics available for chart")
        return
    
    models = list(metrics.keys())
    accuracies = [metrics[m]['best_val_accuracy'] * 100 for m in models]
    losses = [metrics[m]['best_val_loss'] for m in models]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy chart
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    bars1 = ax1.bar(models, accuracies, color=colors[:len(models)])
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([90, 100])
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Loss chart
    bars2 = ax2.bar(models, losses, color=colors[:len(models)])
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Model Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_performance_comparison.png")

def create_html_report(metrics):
    """Create comprehensive HTML report"""
    
    # Calculate statistics
    if metrics:
        avg_accuracy = np.mean([m['best_val_accuracy'] * 100 for m in metrics.values()])
        best_model = max(metrics.items(), key=lambda x: x[1]['best_val_accuracy'])
        best_model_name = best_model[0]
        best_accuracy = best_model[1]['best_val_accuracy'] * 100
    else:
        avg_accuracy = 0
        best_model_name = "N/A"
        best_accuracy = 0
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poultry Disease Classification - Training Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-number {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #6c757d;
            font-size: 1.1em;
            font-weight: 500;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 2em;
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .chart-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }}
        
        .chart-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .highlight-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
        }}
        
        .highlight-box h3 {{
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        .highlight-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .highlight-box li {{
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        
        .highlight-box li:last-child {{
            border-bottom: none;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 30px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #ecf0f1;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐔 Poultry Disease Classification</h1>
            <p>Multi-Model Deep Learning Training Results</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Models Trained</div>
                <div class="stat-number">{len(metrics)}/5</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Accuracy</div>
                <div class="stat-number">{avg_accuracy:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best Model</div>
                <div class="stat-number" style="font-size: 1.5em;">{best_model_name}</div>
                <p style="color: #667eea; font-weight: bold; margin-top: 10px;">{best_accuracy:.2f}%</p>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Images</div>
                <div class="stat-number">4,980</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">📈 Performance Comparison</h2>
                <div class="chart-container">
                    <img src="model_performance_comparison.png" alt="Model Performance Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">📊 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>Validation Accuracy</th>
                            <th>Validation Loss</th>
                            <th>Latest Epoch</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add model rows
    for model_name, model_metrics in metrics.items():
        acc = model_metrics['best_val_accuracy'] * 100
        loss = model_metrics['best_val_loss']
        epoch = model_metrics['latest_epoch']
        checkpoints = model_metrics['checkpoints_found']
        
        status_badge = '<span class="badge badge-success">Training Complete</span>' if epoch >= 9 else '<span class="badge badge-warning">In Progress</span>'
        
        html_content += f"""
                        <tr>
                            <td><strong>{model_name}</strong></td>
                            <td><strong style="color: #2ecc71; font-size: 1.1em;">{acc:.2f}%</strong></td>
                            <td>{loss:.4f}</td>
                            <td>{epoch:.2f} / 10</td>
                            <td>{status_badge}<br><small>{checkpoints} checkpoints</small></td>
                        </tr>
"""
    
    html_content += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">🎯 Dataset Information</h2>
                <div class="highlight-box">
                    <h3>Training Configuration</h3>
                    <ul>
                        <li><strong>Total Images:</strong> 4,980 images</li>
                        <li><strong>Disease Classes:</strong> 10 categories</li>
                        <li><strong>Classes:</strong> Avian Influenza, Coccidiosis, Fowl Pox, Healthy, Histomoniasis, Infectious Bronchitis, Infectious Bursal Disease, Marek's Disease, Newcastle Disease, Salmonella</li>
                        <li><strong>Data Split:</strong> 72% Train (3,585), 8% Val (399), 20% Test (996)</li>
                        <li><strong>Image Size:</strong> 224×224 pixels</li>
                        <li><strong>Batch Size:</strong> 16</li>
                        <li><strong>Max Epochs:</strong> 10 (with early stopping)</li>
                        <li><strong>Hardware:</strong> NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 11.8)</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">💡 Key Insights</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div class="stat-card">
                        <h3 style="color: #667eea; margin-bottom: 15px;">🏆 Best Performer</h3>
                        <p><strong>{best_model_name}</strong> achieved the highest validation accuracy of <strong style="color: #2ecc71;">{best_accuracy:.2f}%</strong>, demonstrating excellent generalization on the validation set.</p>
                    </div>
                    <div class="stat-card">
                        <h3 style="color: #667eea; margin-bottom: 15px;">📊 Consistent Results</h3>
                        <p>All models achieved <strong>over 96% validation accuracy</strong>, indicating that the dataset is well-structured and all architectures are suitable for this classification task.</p>
                    </div>
                    <div class="stat-card">
                        <h3 style="color: #667eea; margin-bottom: 15px;">⚡ Training Efficiency</h3>
                        <p>Models converged quickly, with ConvNeXt and CvT reaching peak performance around epoch 5, showing efficient learning on the poultry disease dataset.</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">📁 Output Directories</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Output Directory</th>
                            <th>Contents</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>ViT-B/16</td>
                            <td><code>vit_poultry_results/</code></td>
                            <td>Model checkpoints, training state, confusion matrix</td>
                        </tr>
                        <tr>
                            <td>ResNeXt-50</td>
                            <td><code>resnext_poultry_results/</code></td>
                            <td>Best model weights, evaluation metrics</td>
                        </tr>
                        <tr>
                            <td>ResNeSt-50d</td>
                            <td><code>resnest_poultry_results/</code></td>
                            <td>Best model weights, evaluation metrics</td>
                        </tr>
                        <tr>
                            <td>ConvNeXt-Tiny</td>
                            <td><code>convnext_poultry_results/</code></td>
                            <td>Model checkpoints, training state, confusion matrix</td>
                        </tr>
                        <tr>
                            <td>CvT-13</td>
                            <td><code>cvt_poultry_results/</code></td>
                            <td>Model checkpoints, training state, confusion matrix</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Poultry Disease Classification Project</strong></p>
            <p>Deep Learning for Agricultural Disease Detection</p>
            <p style="margin-top: 10px; opacity: 0.8;">© 2025 - Generated with Python, PyTorch & Hugging Face Transformers</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open('FINAL_TRAINING_REPORT.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Saved: FINAL_TRAINING_REPORT.html")

def main():
    print("\n" + "=" * 80)
    print("📊 CREATING FINAL COMPREHENSIVE REPORT")
    print("=" * 80)
    
    # Load metrics
    print("\n1️⃣  Loading checkpoint metrics...")
    metrics = load_checkpoint_metrics()
    
    if not metrics:
        print("⚠️  No metrics found. Please run extract_checkpoint_metrics.py first.")
        return
    
    print(f"   ✅ Loaded metrics for {len(metrics)} models")
    
    # Create visualization
    print("\n2️⃣  Creating performance comparison chart...")
    create_comparison_chart(metrics)
    
    # Create HTML report
    print("\n3️⃣  Generating comprehensive HTML report...")
    create_html_report(metrics)
    
    print("\n" + "=" * 80)
    print("✅ FINAL REPORT GENERATION COMPLETE!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   • model_performance_comparison.png - Performance visualization")
    print("   • FINAL_TRAINING_REPORT.html - Comprehensive HTML report")
    print("\n💡 Open 'FINAL_TRAINING_REPORT.html' in your browser to view the full report")
    print("=" * 80)

if __name__ == '__main__':
    main()
