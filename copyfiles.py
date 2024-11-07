import os
import shutil

# Set the dataset name
dataset_name = 'truck'

# Define paths
img_names_file = f'./tandt_db/img_names/{dataset_name}_img_name.txt'
source_dir = f'./tandt_db/tandt/{dataset_name}/images/'
target_dir = f'./tandt_db/eval/{dataset_name}100/input/'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Read image names from the text file
with open(img_names_file, 'r') as file:
    img_names = file.read().splitlines()

# Copy the image files to the target directory
for img_name in img_names:
    source_path = os.path.join(source_dir, f'{img_name}.jpg')
    target_path = os.path.join(target_dir, f'{img_name}.jpg')
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        print(f'Copied {source_path} to {target_path}')
    else:
        print(f'File {source_path} does not exist')

print('Copying completed.')