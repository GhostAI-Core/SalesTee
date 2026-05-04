import os

base_dir = r"C:\Users\garth\.gemini\antigravity\scratch\SalesTee"
for root, dirs, files in os.walk(base_dir):
    if ".git" in dirs: dirs.remove(".git")
    for file in files:
        if file.endswith(".py") or file.endswith(".md") or file.endswith(".json"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = content.replace('training_tee', 'training_tee')
            new_content = new_content.replace('TRAINING_TEE', 'TRAINING_TEE')
            new_content = new_content.replace('TrainingTee', 'TrainingTee')
            
            if new_content != content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
print("done")
