import subprocess
import os

scripts = [
    "train_model.py",
    "train_simple_model.py",
    "train_poultry_disease_vit.py"
]

for script in scripts:
    print(f"🚀 Running {script}...")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ {script} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script} failed with error {e}")
