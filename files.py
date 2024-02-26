import os
import random
import shutil


source_folder = ''

images1 = ''

images2 = ''


image_files = [file for file in os.listdir(source_folder)]


random.shuffle(image_files)


for filename in image_files[0:10]:
    images1_path = os.path.join(source_folder, filename)
    save1_path = os.path.join(images1, filename)
    shutil.copyfile(images1_path, save1_path)

for filename in image_files[10:512]:
    images2_path = os.path.join(source_folder, filename)
    save2_path = os.path.join(images2, filename)
    shutil.copyfile(images2_path, save2_path)
