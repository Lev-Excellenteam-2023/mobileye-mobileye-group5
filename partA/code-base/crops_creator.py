import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image

PADDING_RATIO = 1
IMG_SIZE = (40, 120)

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH, IMAG_PATH

from pandas import DataFrame


def make_crop(coordinates, img_path, tl_diameter, color):
    img = Image.open(img_path)

    x, y = coordinates

    # Adjust for different colors
    if color == "red":
        y += 2 * tl_diameter
    elif color == "green":
        y -= 2 * tl_diameter

    padding_pixel = PADDING_RATIO * tl_diameter
    width = tl_diameter
    height = 8 * tl_diameter

    x0, x1 = x - width - padding_pixel, x + width + padding_pixel
    y0, y1 = y - height / 1.8 - padding_pixel, y + height / 2 + padding_pixel

    # Ensure cropping coordinates don't exceed the image frame
    x0, x1 = max(0, x0), min(img.width, x1)
    y0, y1 = max(0, y0), min(img.height, y1)

    cropped_img = img.crop((x0, y0, x1, y1))
    resized_crop = cropped_img.resize(IMG_SIZE)
    return x0, x1, y0, y1, resized_crop




def check_crop(img_path, x0, x1, y0, y1, threshold=0.75):

    is_tl, ignore = False, False
    json_path = img_path.replace('leftImg8bit.png', 'gtFine_polygons.json')

    with open(json_path, 'r') as file:
        data = json.load(file)

        for obj in data['objects']:
            if obj['label'] == 'traffic light':
                polygon = obj['polygon']
                min_x = min(point[0] for point in polygon)
                max_x = max(point[0] for point in polygon)
                min_y = min(point[1] for point in polygon)
                max_y = max(point[1] for point in polygon)

                # Overlapping box coordinates
                overlap_x0 = max(x0, min_x)
                overlap_x1 = min(x1, max_x)
                overlap_y0 = max(y0, min_y)
                overlap_y1 = min(y1, max_y)

                # Calculate areas
                if overlap_x0 < overlap_x1 and overlap_y0 < overlap_y1:
                    overlap_area = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
                    tl_box_area = (max_x - min_x) * (max_y - min_y)

                    if overlap_area / tl_box_area >= threshold:
                        return True, False

    return is_tl, ignore


def create_crops(df: DataFrame) -> DataFrame:
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    result_df = DataFrame(columns=CROP_RESULT)

    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}

    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        x_coord = row[X]
        y_coord = row[Y]
        imag_path = row[IMAG_PATH]# .replace("\\","/")
        color = row[COLOR]
        tl_diameter = 15  # todo get it from the caller

        x0, x1, y0, y1, crop_content = make_crop((x_coord, y_coord), imag_path, tl_diameter, color)

        crop_name = f"{row['name']}_{index}.png"
        crop_path: Path = CROP_DIR / crop_name  # todo check how the value looklike (os.path.join+name)

        crop_content.save(crop_path)

        result_template[X0] = x0
        result_template[X1] = x1
        result_template[Y0] = y0
        result_template[Y1] = y1
        result_template[COL] = color
        result_template[CROP_PATH] = str(crop_path)

        # todo  implement ignore option
        is_true, ignore = check_crop(imag_path, x0, x1, y0, y1)
        result_template[IS_TRUE] = is_true
        result_template[IGNOR] = ignore

        result_df = result_df._append(result_template, ignore_index=True)
    return result_df
