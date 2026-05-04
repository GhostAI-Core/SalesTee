import os
import re
import shutil

base_dir = r"C:\Users\garth\.gemini\antigravity\scratch\SalesTee"

# 1. Delete meth files
meth_dir = os.path.join(base_dir, "data_store", "methodology")
if os.path.exists(meth_dir):
    shutil.rmtree(meth_dir)
    os.makedirs(meth_dir, exist_ok=True) # Keep the directory empty

# 2. Rename files
for root, dirs, files in os.walk(base_dir):
    if ".git" in dirs:
        dirs.remove(".git")
    for file in files:
        if "training_tee" in file.lower():
            old_path = os.path.join(root, file)
            # Find exact case replacements if possible, else lowercase
            new_name = file.replace("training_tee", "training_tee").replace("Training Tee", "TrainingTee")
            new_path = os.path.join(root, new_name)
            os.rename(old_path, new_path)

# 3. Replace content in files
def replace_content(text):
    # Class names
    text = re.sub(r'TrainingTeeTokenizer', 'TrainingTeeTokenizer', text)
    text = re.sub(r'TrainingTeeEncoder', 'TrainingTeeEncoder', text)
    text = re.sub(r'TrainingTeeDecoder', 'TrainingTeeDecoder', text)
    
    # Constants
    text = re.sub(r'TRAINING_TEE_ROOT', 'TRAINING_TEE_ROOT', text)
    text = re.sub(r'\bTRAINING_TEE\b', 'TRAINING_TEE', text)
    
    # Text mentions
    text = re.sub(r"Training Tee's", "Training Tee's", text)
    text = re.sub(r'\bTrainingTee\b', 'Training Tee', text)
    text = re.sub(r'\btraining_tee\b', 'training_tee', text)
    
    return text

for root, dirs, files in os.walk(base_dir):
    if ".git" in dirs:
        dirs.remove(".git")
    for file in files:
        if file.endswith(".py") or file.endswith(".md") or file.endswith(".txt"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = replace_content(content)
            
            if new_content != content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)

print("Replacement complete.")
