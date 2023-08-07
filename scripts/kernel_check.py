import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def apply_kernel(img_array, kernel):
    processed_image = convolve2d(img_array, kernel, mode='same', boundary='symm')
    return processed_image


def extract_layer(img, index):
    arr = np.asarray(img)
    layer = arr[:, :, index]
    return layer


def highlight_differences(img1_array, img2_array,path_to_org_img = "p2.png"):
    # diff = [ | a1 - b1 | , | a2 - b2 | .....]
    diff = np.abs(img1_array.astype(float) - img2_array.astype(float))

    # shape of sum_diff = [ ( sum of patch 5x5 from diff , ( idx_x,idx_y) , ....... ]
    sum_diffs = []
    # Compare using a patch of 5x5 pixels
    for i in range(diff.shape[0] - 5):
        for j in range(diff.shape[1] - 5):
            patch = diff[i:i + 5, j:j + 5]
            sum_diffs.append(((np.sum(patch)), (i + 2, j + 2)))

    sum_diffs = process_list(sum_diffs)  # it wll delete the closes points
    sorted_diffs = sorted(sum_diffs, key=lambda x: -x[0])
    top_diff_coords = [item[1] for item in sorted_diffs[:5]]  # how many points we want

    #  replace to the correct coordinates
    for i in range(len(top_diff_coords)):
        top_diff_coords[i] = convert_to_original_index(top_diff_coords[i][0], top_diff_coords[i][1], 0.15)

    img = Image.open(path_to_org_img)

    green_layer = extract_layer(img, 1)
    red_layer = extract_layer(img,0)

    drawing_color = 255


    img1_with_markers = Image.fromarray(green_layer.astype(np.uint8))
    draw = ImageDraw.Draw(img1_with_markers)
    for coord in top_diff_coords:
        draw.ellipse([(coord[1] - 30, coord[0] - 30), (coord[1] + 30, coord[0] + 30)], outline=drawing_color, width=10)
    return img1_with_markers


def process_layers(image_path):
    img = Image.open(image_path)
    kernel = create_custom_kernel()

    # Process green layer
    green_layer = extract_layer(img, 1)
    green_layer_processed = apply_kernel(green_layer, kernel)
    highlighted_green = highlight_differences(green_layer, green_layer_processed)

    # Process red layer
    red_layer = extract_layer(img, 0)
    red_layer_processed = apply_kernel(red_layer, kernel)
    highlighted_red = highlight_differences(red_layer, red_layer_processed)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(highlighted_red, cmap='gray')
    plt.title('Red Layer with Differences')

    plt.subplot(1, 2, 2)
    plt.imshow(highlighted_green, cmap='gray')
    plt.title('Green Layer with Differences')

    plt.show()


def create_custom_kernel():
    kernel = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    # kernel = kernel - kernel.mean()
    normalized_kernel = kernel / kernel.sum()

    return normalized_kernel


def bucket_key(coord, threshold=10):
    x, y = coord
    return (x // threshold, y // threshold)


def process_list(lst):
    buckets = {}
    # Step 1: Bucketize each point
    for magnitude, coord in lst:
        key = bucket_key(coord)
        if key not in buckets or buckets[key][0] < magnitude:
            buckets[key] = (magnitude, coord)
    # Step 2 and 3: Get points with highest magnitude from each bucket
    result = [(magnitude, coord) for magnitude, coord in buckets.values()]

    return result


def convert_to_original_index(x_resized, y_resized, scale_factor):
    x_original = int(x_resized / scale_factor)
    y_original = int(y_resized / scale_factor)
    return (x_original, y_original)


if __name__ == '__main__':
    image_path = 'resized_rsz_2tl.png'
    process_layers(image_path)
