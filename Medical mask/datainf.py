
import os
import shutil
folder1 = "Medical Mask/annotations"
folder2 = "Medical Mask/images"

files_in_folder1 = [os.path.splitext(file)[0] for file in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, file))]
files_in_folder2 = os.listdir(folder2)

common_files = [file for file in files_in_folder1 if file in files_in_folder2]


new_folder = "Medical Mask/AnnImages"

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

for file in common_files:
    file2_path = os.path.join(folder2, file )
    new_path = os.path.join(new_folder, file)

    if os.path.isfile(file2_path):
        shutil.copy(file2_path, new_path)