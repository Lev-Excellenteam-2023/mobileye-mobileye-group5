# This file contains the skeleton you can use for traffic light attention
import json
import argparse
import random
from datetime import datetime
from argparse import Namespace
from pathlib import Path
from typing import Sequence, Optional, List, Any, Dict

from matplotlib.axes import Axes

# Internal imports... Should not fail
from consts import IMAG_PATH, JSON_PATH, NAME, SEQ_IMAG, X, Y, COLOR, RED, GRN, DATA_DIR, TFLS_CSV, CSV_OUTPUT, \
    SEQ, CROP_DIR, CROP_CSV_NAME, ATTENTION_RESULT, ATTENTION_CSV_NAME, ZOOM, RELEVANT_IMAGE_PATH, COL, ATTENTION_PATH, \
    CSV_INPUT
from misc_goodies import show_image_and_gt
from data_utils import get_images_metadata
from crops_creator import create_crops

import tqdm  # for the progress bar
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter, label
from PIL import Image
import matplotlib.pyplot as plt

RED_CHANNEL_INDEX = 0
GREEN_CHANNEL_INDEX = 1
BLUE_CHANNEL_INDEX = 2
MIN_RED_RATE = 0.6
MAX_RED_RATE = 0.7
GREEN_THRESHOLD = 0.95


def my_conv2d(image_1d, kernel):
    image_1d = image_1d.astype(float)
    return sg.correlate2d(image_1d, kernel, mode='valid')


def get_max_points(conv_2d, thresh: float = 0.6):
    random_numbers = np.random.uniform(0, 0.0000001, size=conv_2d.shape)

    # Add the random numbers to the image
    conv_2d = conv_2d + random_numbers

    conv_2d = conv_2d.astype(float)
    max_filter = maximum_filter(conv_2d, size=50)

    max_filter[max_filter < thresh] = None

    filter_idx = np.argwhere(max_filter == conv_2d)

    if filter_idx.any():
        x, y = np.array(filter_idx[:, 0]).ravel(), \
            np.array(filter_idx[:, 1]).ravel()
    else:
        x, y = [], []

    return list(x), list(y)


def find_tfl_lights(c_image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement.

    :param c_image: a H*W*3 RGB image of dtype np.uint8 (RGB, 0-255).
    :param kwargs: Whatever you want.
    :return: Dictionary with at least the following keys: 'x', 'y', 'col', each containing a list (same lengths).
    # Note there are no explicit strings in the code-base. ALWAYS USE A CONSTANT VARIABLE INSTEAD!.
    """

    red_channel = c_image[:, :, RED_CHANNEL_INDEX]
    green_channel = c_image[:, :, GREEN_CHANNEL_INDEX]
    blue_channel = c_image[:, :, BLUE_CHANNEL_INDEX]
    red_mask = (red_channel < MIN_RED_RATE) | (blue_channel > red_channel * 0.7) | (green_channel > red_channel * 0.7)
    green_msk = (red_channel > MAX_RED_RATE) | (blue_channel * 0.8 < red_channel) | (green_channel * 0.8 < red_channel)
    red_channel_copy = red_channel.copy()
    red_channel_copy[red_mask] = [0]
    green_channel_copy = green_channel.copy()
    green_channel_copy[green_msk] = [0]
    y_red, x_red = get_max_points(red_channel_copy)
    y_green, x_green = get_max_points(green_channel_copy, GREEN_THRESHOLD)






    return {X: x_red + x_green,
            Y: y_red + y_green,
            COLOR: [RED] * len(x_red) + [GRN] * len(x_green),
            }


def test_find_tfl_lights(row: Series, args: Namespace) -> DataFrame:
    """
    Run the attention code-base
    """
    image_path: str = row[IMAG_PATH]# .replace("\\","/")
    json_path: str = row[JSON_PATH]# .replace("\\","/")
    image: np.ndarray = np.array(Image.open(image_path), dtype=np.float32) / 255

    if args.debug and json_path is not None:
        # This code-base demonstrates the fact you can read the bounding polygons from the json files
        # Then plot them on the image. Try it if you think you want to. Not a must...
        gt_data: Dict[str, Any] = json.loads(Path(json_path).read_text())
        what: List[str] = ['traffic light']
        objects: List[Dict[str, Any]] = [o for o in gt_data['objects'] if o['label'] in what]
        ax: Optional[Axes] = show_image_and_gt(image, objects, f"{row[SEQ_IMAG]}: {row[NAME]} GT")
    else:
        ax = None

    # In case you want, you can pass any parameter to find_tfl_lights, because it uses **kwargs
    attention_dict: Dict[str, Any] = find_tfl_lights(image, some_threshold=42, debug=args.debug)
    attention: DataFrame = pd.DataFrame(attention_dict)

    # Copy all image metadata from the row into the results, so we can track it later
    for k, v in row.items():

        if k == 'x' or k == 'y':
            continue
        attention[k] = v

    tfl_x: np.ndarray = attention[X].values
    tfl_y: np.ndarray = attention[Y].values
    color: np.ndarray = attention[COLOR].values
    is_red = color == RED

    print(f"Image: {image_path}, {is_red.sum()} reds, {len(is_red) - is_red.sum()} greens..")

    if args.debug:
        # And here are some tips & tricks regarding matplotlib
        # They will look like pictures if you use jupyter, and like magic if you use pycharm!
        # You can zoom one image, and the other will zoom accordingly.
        # I think you will find it very very useful!
        plt.figure(f"{row[SEQ_IMAG]}: {row[NAME]} detections")
        plt.clf()
        plt.subplot(211, sharex=ax, sharey=ax)
        plt.imshow(image)
        plt.title('Original image.. Always try to compare your output to it')
        plt.plot(tfl_x[is_red], tfl_y[is_red], 'rx', markersize=4)
        plt.plot(tfl_x[~is_red], tfl_y[~is_red], 'g+', markersize=4)
        # Now let's convolve. Cannot convolve a 3D image with a 2D kernel, so I create a 2D image
        # Note: This image is useless for you, so you solve it yourself
        useless_image: np.ndarray = np.std(image, axis=2)  # No. You don't want this line in your code-base
        highpass_kernel_from_lecture: np.ndarray = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) - 1 / 9
        hp_result: np.ndarray = sg.convolve(useless_image, highpass_kernel_from_lecture, 'same')
        plt.subplot(212, sharex=ax, sharey=ax)
        plt.imshow(hp_result)
        plt.title('Some useless image for you')
        plt.suptitle("When you zoom on one, the other zooms too :-)")

    return attention


def prepare_list(in_csv_file: Path, args: Namespace) -> DataFrame:
    """
    We assume all students are working on the same CSV with files.
    This filters the list, so if you want to test some specific images, it's easy.
    This way, you can ask your friends how they performed on image 42 for example
    You may want to set the default_csv_file to anything on your computer, to spare the -f parameter.
    Note you will need different CSV files for attention and NN parts.
    The CSV must have at least columns: SEQ, NAME, TRAIN_TEST_VAL.
    """
    if args.image is not None:
        # Don't limit by count, take explicit images
        args.count = None

    csv_list: DataFrame = get_images_metadata(in_csv_file,
                                              max_count=args.count,
                                              take_specific=args.image)
    return pd.concat([pd.DataFrame(columns=CSV_INPUT), csv_list], ignore_index=True)


def run_on_list(meta_table: pd.DataFrame, func: callable, args: Namespace) -> pd.DataFrame:
    """
    Take a function, and run it on a list. Return accumulated results.

    :param meta_table: A DF with the columns your function requires
    :param func: A function to take a row of the DF, and return a DF with some results
    :param args:
    """
    acc: List[DataFrame] = []
    time_0: datetime = datetime.now()
    for _, row in tqdm.tqdm(meta_table.iterrows()):
        res: DataFrame = func(row, args)
        acc.append(res)
    time_1: datetime = datetime.now()
    all_results: DataFrame = pd.concat(acc).reset_index(drop=True)
    print(f"Took me {(time_1 - time_0).total_seconds()} seconds for "
          f"{len(all_results)} results from {len(meta_table)} files")

    return all_results


def save_df_for_part_2(crops_df: DataFrame, results_df: DataFrame):
    if not ATTENTION_PATH.exists():
        ATTENTION_PATH.mkdir()

    # Order the df by sequence, a nice to have.
    crops_sorted: DataFrame = crops_df.sort_values(by=SEQ)
    results_sorted: DataFrame = results_df.sort_values(by=SEQ_IMAG)

    attention_df: DataFrame = DataFrame(columns=ATTENTION_RESULT)
    row_template: Dict[str, Any] = {RELEVANT_IMAGE_PATH: '', X: '', Y: '', ZOOM: 0, COL: ''}
    for index, row in results_sorted.iterrows():
        row_template[RELEVANT_IMAGE_PATH] = row[IMAG_PATH]
        row_template[X], row_template[Y] = row[X], row[Y]
        row_template[COL] = row[COLOR]
        attention_df = attention_df._append(row_template, ignore_index=True)
    attention_df.to_csv(ATTENTION_PATH / ATTENTION_CSV_NAME, index=False)
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def parse_arguments(argv: Optional[Sequence[str]]):
    """
    Here are all the arguments in the attention stage.
    """
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=int, nargs='+', help='Specific image number(s) to run')
    parser.add_argument("-c", "--count", type=int, default=300, help="Max images to run")
    parser.add_argument('-f', '--in_csv_file', type=str, help='CSV file to read')
    parser.add_argument('-nd', '--no_debug', action='store_true', help='Show debug info')
    parser.add_argument('--attention_csv_file', type=str, help='CSV to write results to')

    args = parser.parse_args(argv)

    args.debug = not args.no_debug

    return args


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    args: Namespace = parse_arguments(argv)
    default_csv_file: Path = DATA_DIR / TFLS_CSV
    csv_filename: Path = Path(args.in_csv_file) if args.in_csv_file else default_csv_file

    # This is your output CSV, look for CSV_OUTPUT const to see its columns.
    # No need to touch this function, if your curious about the result, put a break point and look at the result
    meta_table: DataFrame = prepare_list(csv_filename, args)
    print(f"About to run attention on {len(meta_table)} images. Watch out there!")

    # When you run your find find_tfl_lights, you want to add each time the output (x,y coordinates, color etc.)
    # to the output Dataframe, look at CSV_OUTPUT to see the names of the column and in what order.
    all_results: DataFrame = run_on_list(meta_table, test_find_tfl_lights, args)
    combined_df: DataFrame = pd.concat([pd.DataFrame(columns=CSV_OUTPUT), all_results], ignore_index=True)

    # make crops out of the coordinates from the DataFrame
    crops_df: DataFrame = create_crops(combined_df)

    # save the DataFrames in the right format for stage two.
    save_df_for_part_2(crops_df, combined_df)
    print(f"Got a total of {len(combined_df)} results")

    if args.debug:
        plt.show(block=True)


if __name__ == '__main__':
    main()
