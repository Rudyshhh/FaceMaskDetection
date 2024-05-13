import shutil
import csv
import os

# Set up source and destination directories
src_dir = 'Medical Mask/AnnImages'
with_mask_dir = 'Medical Mask/with_mask'
without_mask_dir = 'Medical Mask/no_mask'

# Create destination directories if they don't exist
os.makedirs(with_mask_dir, exist_ok=True)
os.makedirs(without_mask_dir, exist_ok=True)

# Open CSV file and read rows
with open('MyTrain.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Copy image to with_mask or without_mask directory based on maskValue
        if row['maskValue'] == 'face_with_mask':
            shutil.copy(os.path.join(src_dir, row['name']), os.path.join(with_mask_dir, row['name']))
        if row['maskValue'] == 'face_no_mask':
            shutil.copy(os.path.join(src_dir, row['name']), os.path.join(without_mask_dir, row['name']))