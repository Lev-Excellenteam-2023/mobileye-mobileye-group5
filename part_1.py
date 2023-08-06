import os
from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import signal as sg
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import label

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = os.path.join('.', 'tl_photos')

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]



def create_binary_mask_from_indices(shape, indices_list):
    # Create a binary mask with ones at the specified indices and zeros elsewhere
    binary_mask = np.zeros(shape, dtype=np.int32)
    binary_mask[tuple(zip(*indices_list))] = 1

    return binary_mask
def keep_one_maximum_per_component(input_array ):
    # Thresholding

    s = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    # Perform connected component labeling
    labeled_array, num_features = label(input_array, structure=s)


    for f in range(1,num_features+1):
         x_indices, y_indices  = np.where(labeled_array == f)
         labeled_array[(x_indices[:-1],y_indices[:-1])]=0

    return labeled_array



def my_conv2d(image_1d,kernel):


    image_1d = image_1d.astype(float)

    return sg.correlate2d(image_1d, kernel, mode='valid')


def find_tfl_lights(c_image: np.ndarray,
                    **kwargs) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement.

    :param c_image: The image itself as np.uint8, shape of (H, W, 3).
    :param kwargs: Whatever config you want to pass in here.
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """

    kernel = np.array(     [-2,
                            -8,
                            0,
                            0,
                            0,
                            1
                            ,1
                            ,2]).reshape(8,1)


    kernel2 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    filter_green_c = my_conv2d(c_image[:,:,1], kernel2)

    conv_red = my_conv2d(filter_green_c,kernel)

   ######## maximum filter
    max_filter_red = maximum_filter(conv_red, size=5)
    max_filter_red[max_filter_red < 800] = None
    filter_red_idx = np.argwhere(max_filter_red == conv_red)

   # filter_red_idx = np.array([idx for idx in t_filter_red_idx if idx in max_filter_red])

    max_mask = create_binary_mask_from_indices(conv_red.shape,filter_red_idx)

    max_mask_one_component = keep_one_maximum_per_component(max_mask)

    filter_red_idx = np.argwhere(max_mask_one_component != 0)

    print(filter_red_idx)

    if filter_red_idx.any():
        red_x , red_y = np.array(filter_red_idx[:, 0]).ravel() , np.array(filter_red_idx[:, 1]).ravel()
    else:
        red_x, red_y = [], []
        print('max empty')

        #print(red_x)
    #print(red_y)
    return red_y , red_x, [600, 800], [400, 300]




def show_two_images(image, result):
    """
    Displays two 2D NumPy arrays as images side by side using matplotlib.

    Parameters:
        array1 (numpy.ndarray): The first 2D array to be displayed as an image.
        array2 (numpy.ndarray): The second 2D array to be displayed as an image.
    """

    # Plot the original image and the convolved result
    plt.figure(figsize=(8, 4))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot the convolved result
    plt.subplot(1, 2, 2)
    plt.imshow(result,cmap="gray")
    plt.title('Convolved Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    # ensure a fresh canvas for plotting the image and objects.
    plt.figure(fig_num).clf()
    # displays the input image.
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            # Extract the 'polygon' array from the image object
            poly: np.array = np.array(image_object['polygon'])
            # Use advanced indexing to create a closed polygon array
            # The modulo operation ensures that the array is indexed circularly, closing the polygon
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            # gets the x coordinates (first column -> 0) anf y coordinates (second column -> 1)
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'r'
            plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
            plt.legend()


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, fig_num=None):
    """
    Run the attention code.
    """
    # using pillow to load the image
    image: Image = Image.open(image_path)
    # converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)

    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)
    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    # If you entered a custom dir to run from or the default dir exist in your project then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        # gets a list of all the files in the directory that ends with "_leftImg8bit.png".
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))

        for image in file_list:
            # Convert the Path object to a string using as_posix() method
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            test_find_tfl_lights(image_path, image_json_path)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
