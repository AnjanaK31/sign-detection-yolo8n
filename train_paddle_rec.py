import os
import shutil
import sys
from paddlex.utils.config import get_config
from paddlex import build_dataset_checker, build_trainer

# 1. Update dict.txt in dataset directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "paddle_rec_dataset")
my_chars_path = os.path.join(dataset_dir, "my_chars.txt")
dict_path = os.path.join(dataset_dir, "dict.txt")

print(f"Ensuring character dictionary is copied to dict.txt...")
if os.path.exists(my_chars_path):
    shutil.copy2(my_chars_path, dict_path)
    print(f"Copied {my_chars_path} -> {dict_path}")
else:
    print(f"ERROR: {my_chars_path} not found!")
    sys.exit(1)

# 2. Load configuration
config_path = os.path.join(dataset_dir, "rec_finetune.yaml")
print(f"Loading configuration from {config_path}...")
config = get_config(config_path)

# 3. Verify dataset
print("Building dataset checker...")
checker = build_dataset_checker(config)
print("Running dataset check...")
check_res = checker.check()
print("Dataset Check Results:", check_res)

if not check_res.get("check_pass", False):
    print("ERROR: Dataset validation failed!")
    sys.exit(1)
print("SUCCESS: Dataset validation passed!")

# 4. Run training
print("Building trainer...")
trainer = build_trainer(config)
print("Launching training (5 epochs on CPU)...")
trainer.train()
print("Training completed!")
