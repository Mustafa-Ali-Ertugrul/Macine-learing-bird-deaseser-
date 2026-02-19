"""
Comprehensive Poultry Disease Dataset Report Generator
Generates detailed HTML and text reports about the dataset
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json

def analyze_csv_data():
    """Analyze CSV metadata if exists"""
    csv_files = ["poultry_labeled_12k.csv", "poultry_labeled.csv", "dataset.csv"]
    
    for csv_file in csv_files:
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            
            disease_counts = df['disease'].value_counts() if 'disease' in df.columns else {}
            
            return {
                'csv_file': csv_file,
                'total_records': len(df),
                'disease_counts': disease_counts.to_dict(),
                'columns': list(df.columns),
                'exists': True
            }
    
    return {'exists': False}

def analyze_image_directories():
    """Analyze actual image files on disk"""
    results = {}
    
    # Check different possible dataset directories
    directories = [
        "poultry_microscopy",
        "organized_poultry_dataset",
        "dataset",
        "data",
        "images"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            results[dir_name] = analyze_directory(dir_path)
    
    return results

def analyze_directory(dir_path):
    """Analyze a specific directory for images"""
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(dir_path.rglob(ext)))
    
    # Count by subdirectory
    disease_counts = defaultdict(int)
    
    for img in all_images:
        # Get the immediate parent directory name
        parent = img.parent.name
        disease_counts[parent] += 1
    
    return {
        'total_images': len(all_images),
        'disease_breakdown': dict(disease_counts),
        'path': str(dir_path)
    }

def analyze_training_results():
    """Analyze training results if available"""
    results_files = [
        "checkpoint_metrics.json",
        "training_results_summary.json",
        "all_models_training_results.json"
    ]
    
    for result_file in results_files:
        if Path(result_file).exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                return {
                    'file': result_file,
                    'data': data,
                    'exists': True
                }
    
    return {'exists': False}

def generate_text_report(csv_data, image_data, training_data):
    """Generate detailed text report"""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE POULTRY DISEASE DATASET REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # CSV Data Section
    report.append("\n📊 METADATA ANALYSIS")
    report.append("-" * 80)
    if csv_data['exists']:
        report.append(f"\nCSV File: {csv_data['csv_file']}")
        report.append(f"Total Records: {csv_data['total_records']:,}")
        report.append(f"Columns: {', '.join(csv_data['columns'])}")
        
        if csv_data['disease_counts']:
            report.append(f"\nDisease Distribution:")
            report.append(f"{'Category':<30} {'Count':>10}  {'Percentage':>10}")
            report.append(f"{'-'*30} {'-'*10}  {'-'*10}")
            
            total = csv_data['total_records']
            for disease, count in sorted(csv_data['disease_counts'].items(), key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100
                report.append(f"{disease:<30} {count:>10,}  {pct:>9.1f}%")
            
            report.append(f"{'-'*30} {'-'*10}  {'-'*10}")
            report.append(f"{'TOTAL':<30} {total:>10,}  {100.0:>9.1f}%")
    else:
        report.append("\n⚠️  No CSV metadata file found")
    
    # Image Files Section
    report.append("\n\n📁 IMAGE FILES ON DISK")
    report.append("-" * 80)
    
    if image_data:
        total_all_images = 0
        
        for dir_name, data in image_data.items():
            report.append(f"\n📂 {dir_name}/")
            report.append(f"   Path: {data['path']}")
            report.append(f"   Total Images: {data['total_images']:,}")
            total_all_images += data['total_images']
            
            if data['disease_breakdown']:
                report.append(f"\n   Disease/Category Breakdown:")
                report.append(f"   {'Category':<30} {'Count':>10}  {'Percentage':>10}")
                report.append(f"   {'-'*30} {'-'*10}  {'-'*10}")
                
                total = data['total_images']
                for category, count in sorted(data['disease_breakdown'].items(), key=lambda x: x[1], reverse=True):
                    pct = (count / total) * 100 if total > 0 else 0
                    report.append(f"   {category:<30} {count:>10,}  {pct:>9.1f}%")
                
                report.append(f"   {'-'*30} {'-'*10}  {'-'*10}")
                report.append(f"   {'TOTAL':<30} {total:>10,}  {100.0:>9.1f}%")
        
        report.append(f"\n🎯 GRAND TOTAL IMAGES: {total_all_images:,}")
    else:
        report.append("\n⚠️  No image directories found")
    
    # Training Results Section
    report.append("\n\n🤖 TRAINING RESULTS")
    report.append("-" * 80)
    
    if training_data['exists']:
        report.append(f"\nResults File: {training_data['file']}")
        data = training_data['data']
        
        if isinstance(data, dict):
            for model_name, metrics in data.items():
                report.append(f"\n{model_name}:")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report.append(f"   {key}: {value}")
                        else:
                            report.append(f"   {key}: {str(value)[:100]}")
    else:
        report.append("\n⚠️  No training results found")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def generate_html_report(csv_data, image_data, training_data):
    """Generate HTML report with visualizations"""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poultry Disease Dataset Report</title>
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
            border-radius: 15px;
            box-shadow: 0 10px 50px rgba(0,0,0,0.3);
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
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-card .label {{
            font-size: 1em;
            opacity: 0.9;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
        
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐔 Poultry Disease Dataset Report</h1>
            <div class="subtitle">Comprehensive Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="content">
"""
    
    # Statistics Overview
    total_images = sum(data['total_images'] for data in image_data.values()) if image_data else 0
    total_csv = csv_data['total_records'] if csv_data['exists'] else 0
    num_classes = len(csv_data['disease_counts']) if csv_data['exists'] else 0
    
    html += f"""
            <div class="section">
                <h2 class="section-title">📊 Overview Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Total Images</div>
                        <div class="number">{total_images:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">CSV Records</div>
                        <div class="number">{total_csv:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Number of Classes</div>
                        <div class="number">{num_classes}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Dataset Directories</div>
                        <div class="number">{len(image_data)}</div>
                    </div>
                </div>
            </div>
"""
    
    # CSV Metadata Section
    if csv_data['exists']:
        html += f"""
            <div class="section">
                <h2 class="section-title">📄 CSV Metadata Analysis</h2>
                <div class="success">
                    <strong>✅ CSV File Found:</strong> {csv_data['csv_file']}
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th>Disease Category</th>
                            <th>Count</th>
                            <th>Percentage</th>
                            <th>Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for disease, count in sorted(csv_data['disease_counts'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_csv) * 100
            html += f"""
                        <tr>
                            <td><strong>{disease}</strong></td>
                            <td>{count:,}</td>
                            <td>{pct:.1f}%</td>
                            <td>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {pct}%"></div>
                                </div>
                            </td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
"""
    else:
        html += """
            <div class="section">
                <h2 class="section-title">📄 CSV Metadata Analysis</h2>
                <div class="warning">
                    <strong>⚠️ No CSV metadata file found</strong>
                </div>
            </div>
"""
    
    # Image Directories Section
    if image_data:
        for dir_name, data in image_data.items():
            html += f"""
            <div class="section">
                <h2 class="section-title">📁 {dir_name}</h2>
                <p><strong>Path:</strong> {data['path']}</p>
                <p><strong>Total Images:</strong> {data['total_images']:,}</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Count</th>
                            <th>Percentage</th>
                            <th>Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            
            for category, count in sorted(data['disease_breakdown'].items(), key=lambda x: x[1], reverse=True):
                pct = (count / data['total_images']) * 100 if data['total_images'] > 0 else 0
                html += f"""
                        <tr>
                            <td><strong>{category}</strong></td>
                            <td>{count:,}</td>
                            <td>{pct:.1f}%</td>
                            <td>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {pct}%"></div>
                                </div>
                            </td>
                        </tr>
"""
            
            html += """
                    </tbody>
                </table>
            </div>
"""
    
    # Training Results Section
    if training_data['exists']:
        html += f"""
            <div class="section">
                <h2 class="section-title">🤖 Training Results</h2>
                <div class="success">
                    <strong>✅ Results Found:</strong> {training_data['file']}
                </div>
                <pre style="background: #f8f9fa; padding: 20px; border-radius: 5px; overflow-x: auto;">
{json.dumps(training_data['data'], indent=2)[:2000]}
                </pre>
            </div>
"""
    
    html += """
        </div>
        
        <div class="footer">
            <p>Generated by Poultry Disease Dataset Report Generator</p>
            <p>Powered by Python & Machine Learning</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html

def main():
    print("\n" + "=" * 80)
    print("🐔 GENERATING COMPREHENSIVE DATASET REPORT")
    print("=" * 80)
    
    # Analyze all data sources
    print("\n1️⃣  Analyzing CSV metadata...")
    csv_data = analyze_csv_data()
    
    print("2️⃣  Analyzing image directories...")
    image_data = analyze_image_directories()
    
    print("3️⃣  Analyzing training results...")
    training_data = analyze_training_results()
    
    # Generate text report
    print("\n4️⃣  Generating text report...")
    text_report = generate_text_report(csv_data, image_data, training_data)
    
    with open('DATASET_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print("   ✅ Saved: DATASET_REPORT.txt")
    
    # Generate HTML report
    print("5️⃣  Generating HTML report...")
    html_report = generate_html_report(csv_data, image_data, training_data)
    
    with open('DATASET_REPORT.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print("   ✅ Saved: DATASET_REPORT.html")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print(text_report)
    
    print("\n" + "=" * 80)
    print("✅ REPORT GENERATION COMPLETE!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   • DATASET_REPORT.txt - Detailed text report")
    print("   • DATASET_REPORT.html - Interactive HTML report")
    print("\n💡 Open 'DATASET_REPORT.html' in your browser to view the interactive report")
    print("=" * 80)

if __name__ == "__main__":
    main()
