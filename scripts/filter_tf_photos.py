import os
import json
import shutil


photo_dir =os.path.join('..', 'photos')
tl_photo_dir  = os.path.join('..', 'tl_photos')

if not os.path.exists(tl_photo_dir):
    os.makedirs(tl_photo_dir)

for filename in os.listdir(photo_dir):
    if filename.endswith(".png"):
        json_file_path = os.path.join(photo_dir, filename.replace('_leftImg8bit.png', '_gtFine_polygons.json'))
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            for obj in data["objects"]:
                if obj["label"] == "traffic light":
                    shutil.move(os.path.join(photo_dir, filename), os.path.join(tl_photo_dir, filename))
                    break
