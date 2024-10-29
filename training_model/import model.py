import kagglehub
import shutil
import os

# Download dataset to default location
path = kagglehub.dataset_download("debashishsau/aslamerican-sign-language-aplhabet-dataset")

# Define target directory
target_dir = "D:/datasets"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Move all files to the target directory
for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), target_dir)

print(f"Files moved to: {target_dir}")
