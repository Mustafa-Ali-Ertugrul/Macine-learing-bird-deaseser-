# Run all training scripts sequentially

Write-Host "Starting ResNeXt Training..."
python train_resnext.py
if ($LASTEXITCODE -ne 0) { Write-Error "ResNeXt failed"; exit 1 }

Write-Host "Starting ResNeSt Training..."
python train_resnest.py
if ($LASTEXITCODE -ne 0) { Write-Error "ResNeSt failed"; exit 1 }

Write-Host "Starting ConvNeXt Training..."
python train_convnext.py
if ($LASTEXITCODE -ne 0) { Write-Error "ConvNeXt failed"; exit 1 }

Write-Host "Starting CvT Training..."
python train_cvt.py
if ($LASTEXITCODE -ne 0) { Write-Error "CvT failed"; exit 1 }

Write-Host "All models trained successfully!"
