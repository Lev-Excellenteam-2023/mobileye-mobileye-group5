import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def resize_image_by_scale(image_path, scale_factor, output_path=""):
    img = Image.open(image_path)
    width, height = img.size
    return img.resize((int(width * scale_factor), int(height * scale_factor)))


def convert_to_original_index(x_resized, y_resized, scale_factor):
    x_original = int(x_resized / scale_factor)
    y_original = int(y_resized / scale_factor)
    return (x_original, y_original)


def apply_kernel(img_array, kernel):
    return convolve2d(img_array, kernel, mode='same', boundary='symm')



def extract_layer(img, index):
    return np.asarray(img)[:, :, index]



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


#  if key is 10 then wll have idxs like [ (10,10),(20,20),..]
#  wll choose as threshold the min distance between 2 entities
def bucket_key(coord, threshold=30):
    x, y = coord
    return (x // threshold, y // threshold)


def process_list(lst):
    buckets = {}
    for magnitude, coord in lst:
        key = bucket_key(coord)
        if key not in buckets or buckets[key][0] < magnitude:
            buckets[key] = (magnitude, coord)
    result = [(magnitude, coord) for magnitude, coord in buckets.values()]
    return result


def highlight_differences(img1, img2, original_image):
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    diff = np.abs(img1_array.astype(float) - img2_array.astype(float))

    sum_diffs = [(np.sum(diff[i:i + 5, j:j + 5]), (i + 2, j + 2))
                 for i in range(diff.shape[0] - 5) for j in range(diff.shape[1] - 5)]

    sum_diffs = process_list(sum_diffs)
    top_diff_coords = [item[1] for item in sorted(sum_diffs, key=lambda x: -x[0])[:10]]

    for i in range(len(top_diff_coords)):
        top_diff_coords[i] = convert_to_original_index(*top_diff_coords[i], 0.15)

    draw = ImageDraw.Draw(original_image)
    for coord in top_diff_coords:
        draw.ellipse([(coord[1] - 30, coord[0] - 30), (coord[1] + 30, coord[0] + 30)], outline=255, width=10)

    return original_image


def process_layers(image_path):
    kernel = create_custom_kernel()

    img = Image.open(image_path)
    original_red = extract_layer(img, 0)
    original_green = extract_layer(img, 1)

    img_resized = resize_image_by_scale(image_path, 0.15)
    red_resized = extract_layer(img_resized, 0)
    green_resized = extract_layer(img_resized, 1)

    red_processed = apply_kernel(red_resized, kernel)
    green_processed = apply_kernel(green_resized, kernel)

    highlighted_red = highlight_differences(red_resized, red_processed, Image.fromarray(original_red))
    highlighted_green = highlight_differences(green_resized, green_processed, Image.fromarray(original_green))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(highlighted_red, cmap='gray')
    plt.title('Red Layer with Differences')

    plt.subplot(1, 2, 2)
    plt.imshow(highlighted_green, cmap='gray')
    plt.title('Green Layer with Differences')

    plt.show()


if __name__ == '__main__':
    image_path = '../local_tests/rsz_2tl.jpg'
    process_layers(image_path)