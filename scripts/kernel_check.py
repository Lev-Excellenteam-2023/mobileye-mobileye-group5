import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.spatial import KDTree


def apply_kernel(img_array, kernel):
    processed_image = convolve2d(img_array, kernel, mode='same', boundary='symm')
    return processed_image


def extract_layer(img, index):
    arr = np.asarray(img)
    layer = arr[:, :, index]
    return layer


def highlight_differences(img1_array, img2_array):
    diff = np.abs(img1_array.astype(float) - img2_array.astype(float))
    sum_diffs = []

    # Compare using a patch of 5x5 pixels
    for i in range(diff.shape[0] - 5):
        for j in range(diff.shape[1] - 5):
            patch = diff[i:i + 5, j:j + 5]
            sum_diffs.append( (abs((np.sum(patch))), (i + 2, j + 2)))

    # Get coordinates of top 10 differences
    sorted_diffs = sorted(sum_diffs, key=lambda x: -x[0])

    sorted_diffs = process_list(sorted_diffs)
    top_diff_coords = [item[1] for item in sorted_diffs[:10]]
    print(top_diff_coords)

    drawing_color =  0

    # Draw markers on the first image array
    img1_with_markers = Image.fromarray(img1_array.astype(np.uint8))
    draw = ImageDraw.Draw(img1_with_markers)
    for coord in top_diff_coords:
        draw.ellipse([(coord[1] - 10, coord[0] - 10), (coord[1] + 10, coord[0] + 10)], outline=drawing_color, width=5)
        print(f"Drawing at coordinates: {coord}")  # Debugging line

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
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 6, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0]
    ])
    # kernel = kernel - kernel.mean()
    # kernel = kernel - kernel.sum()/kernel
    return kernel






def bucket_key(coord, threshold=15):
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
    result = sorted([(magnitude, coord) for magnitude, coord in buckets.values()])

    return result


# [ (magnitude (x,y),.....]
if __name__ == '__main__':
    image_path ='rsz_2tl.jpg'
    process_layers(image_path)
