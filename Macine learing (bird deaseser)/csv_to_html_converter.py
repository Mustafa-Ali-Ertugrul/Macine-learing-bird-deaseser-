import pandas as pd
import os
import sys
from pathlib import Path

def convert_csv_to_html(csv_path, output_html_path):
    """
    Convert CSV file with poultry histopathology images to HTML labeling interface
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create HTML content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poultry Histopathology Labeling Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        .image-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 15px;
            width: 300px;
            text-align: center;
        }
        .image-card img {
            max-width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 5px;
        }
        .image-info {
            margin: 10px 0;
            text-align: left;
        }
        .label-select {
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .controls {
            margin: 20px 0;
            text-align: center;
        }
        .save-btn {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .save-btn:hover {
            background-color: #45a049;
        }
        .counter {
            margin: 10px 0;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐔 Kanatlı Patoloji Etiketleme Aracı</h1>
            <p>Histopatolojik Görüntü Sınıflandırma</p>
        </div>
        
        <div class="counter">
            Etiketlenen: <span id="labeled-count">0</span> / <span id="total-count">0</span>
        </div>
        
        <div class="controls">
            <button class="save-btn" onclick="saveLabels()">Etiketleri Kaydet</button>
        </div>
        
        <div class="image-container" id="image-container">
"""
    
    # Add image cards for each entry
    for index, row in df.iterrows():
        image_path = row['image_path'].replace('\\', '/')
        filename = row['filename']
        width = row['width']
        height = row['height']
        source = row['source']
        disease = row['disease'] if pd.notna(row['disease']) else 'unknown'
        tissue = row['tissue'] if pd.notna(row['tissue']) else 'unknown'
        magnification = row['magnification'] if pd.notna(row['magnification']) else 'unknown'
        
        html_content += f"""
            <div class="image-card" data-index="{index}">
                <img src="{image_path}" alt="{filename}" onerror="this.onerror=null; this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWbvueJh+WKoTwvdGV4dD48L3N2Zz4=';">
                <div class="image-info">
                    <strong>{filename}</strong><br>
                    Boyut: {width}×{height}<br>
                    Kaynak: {source}<br>
                    Mevcut Etiket: {disease}<br>
                    Doku: {tissue} | Büyütme: {magnification}
                </div>
                <select class="label-select" onchange="updateLabel(this)">
                    <option value="unknown" {'selected' if disease == 'unknown' else ''}>Seçiniz...</option>
                    <option value="healthy" {'selected' if disease == 'healthy' else ''}>Sağlıklı (Healthy)</option>
                    <option value="ib" {'selected' if disease == 'ib' else ''}>IB (Infectious Bronchitis)</option>
                    <option value="ibd" {'selected' if disease == 'ibd' else ''}>IBD (Infectious Bursal Disease)</option>
                    <option value="coccidiosis" {'selected' if disease == 'coccidiosis' else ''}>Coccidiosis</option>
                    <option value="salmonella" {'selected' if disease == 'salmonella' else ''}>Salmonella</option>
                    <option value="fatty_liver" {'selected' if disease == 'fatty_liver' else ''}>Fatty Liver Syndrome</option>
                    <option value="histomoniasis" {'selected' if disease == 'histomoniasis' else ''}>Histomoniasis</option>
                </select>
            </div>
"""
    
    # Close HTML content
    html_content += """
        </div>
        
        <div class="controls">
            <button class="save-btn" onclick="saveLabels()">Etiketleri Kaydet</button>
        </div>
    </div>

    <script>
        let labeledCount = 0;
        const totalCount = document.querySelectorAll('.image-card').length;
        
        // Update counters
        document.getElementById('total-count').textContent = totalCount;
        updateLabeledCounter();
        
        function updateLabel(selectElement) {
            const card = selectElement.closest('.image-card');
            card.style.border = selectElement.value !== 'unknown' ? '3px solid #4CAF50' : 'none';
            updateLabeledCounter();
        }
        
        function updateLabeledCounter() {
            const selects = document.querySelectorAll('.label-select');
            labeledCount = Array.from(selects).filter(select => select.value !== 'unknown').length;
            document.getElementById('labeled-count').textContent = labeledCount;
        }
        
        function saveLabels() {
            const labels = [];
            const cards = document.querySelectorAll('.image-card');
            
            cards.forEach(card => {
                const index = card.getAttribute('data-index');
                const select = card.querySelector('.label-select');
                const label = select.value;
                labels.push({index: parseInt(index), label: label});
            });
            
            // In a real implementation, you would send this data to a server
            // For now, we'll just show an alert with the data
            alert('Etiketler kaydedildi!\\n\\n' + JSON.stringify(labels, null, 2));
            
            // In a real implementation, you would also update the CSV file here
            console.log('Labels to save:', labels);
        }
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML labeling interface created: {output_html_path}")
    print(f"Total images: {len(df)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_html_converter.py <input_csv_path> <output_html_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    html_path = sys.argv[2]
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    convert_csv_to_html(csv_path, html_path)