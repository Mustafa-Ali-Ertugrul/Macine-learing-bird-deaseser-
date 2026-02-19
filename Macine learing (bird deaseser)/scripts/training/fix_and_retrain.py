import subprocess
import os

scripts = [
    "train_model.py",       # Will now save to ..._resnet.pth
    "train_simple_model.py" # Will now save to ..._simple.pth
]

print("🚀 Re-training ResNet and SimpleCNN with fixed filenames...")
processes = []

for script in scripts:
    print(f"Starting {script}...")
    # Run in parallel using Popen
    p = subprocess.Popen(["python", script])
    processes.append((script, p))

for script, p in processes:
    p.wait()
    if p.returncode == 0:
        print(f"✅ {script} finished.")
    else:
        print(f"❌ {script} failed.")
