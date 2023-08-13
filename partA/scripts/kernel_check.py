import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

RESIZE_FACTOR = 1  # 0.8 , 033
KERNEL_HEIGHT = 15
KERNEL_WIDTH = 6
BUCKET_THRESHOLD = 20
NUM_OF_RES = 5


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
    green_kernel = np.array([
        [0, -1, -1, -1, 0, 0],
        [0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0],
        [0, 30, 30, 30, 30, 0],
        [10, 30, 50, 30, 10, 0],
        [100, 150, 255, 255, 150, 50],
        [150, 255, 255, 255, 255, 150],
        [100, 200, 255, 255, 200, 100],
        [30, 150, 200, 200, 150, 30],
        [10, 50, 60, 60, 50, 10],
        [5, 30, 30, 30, 30, 5],
        [0, 5, 10, 10, 5, 0]
    ])

    red_kernel = np.flipud(green_kernel)

    # kernel = kernel - kernel.mean()
    normalized_green_kernel = green_kernel / green_kernel.sum()
    normalized_red_kernel = red_kernel / red_kernel.sum()

    return normalized_green_kernel


#  if key is 10 then wll have idxs like [ (10,10),(20,20),..]
#  wll choose as threshold the min distance between 2 entities
def bucket_key(coord):
    x, y = coord
    return (x // BUCKET_THRESHOLD, y // BUCKET_THRESHOLD)


def reduce_list(lst):
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

    half_height = KERNEL_HEIGHT // 2
    half_width = KERNEL_WIDTH // 2


    sum_diffs = [
        (np.sum(diff[i:i + KERNEL_HEIGHT, j:j + KERNEL_WIDTH]),
         (i + half_height, j + half_width))
        for i in range(diff.shape[0] - KERNEL_HEIGHT)
        for j in range(diff.shape[1] - KERNEL_WIDTH)
    ]

    sum_diffs = reduce_list(sum_diffs)
    top_diff_coords = [item[1] for item in sorted(sum_diffs, key=lambda x: -x[0])[:NUM_OF_RES]]

    for i in range(len(top_diff_coords)):
        top_diff_coords[i] = convert_to_original_index(*top_diff_coords[i], RESIZE_FACTOR)

    draw = ImageDraw.Draw(original_image)
    for coord in top_diff_coords:
        draw.ellipse([(coord[1] - 30, coord[0] - 30), (coord[1] + 30, coord[0] + 30)], outline=255, width=10)

    return original_image


def process_layers(image_path):
    kernel = create_custom_kernel()

    img = Image.open(image_path)
    original_red = extract_layer(img, 0)
    original_green = extract_layer(img, 1)

    img_resized = resize_image_by_scale(image_path, RESIZE_FACTOR)
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


def show_kernel(kernel):
    # Ensure kernel is either 2D or 3D with 3 channels
    if len(kernel.shape) not in [2, 3] or (len(kernel.shape) == 3 and kernel.shape[2] != 3):
        raise ValueError("Kernel should be 2D (H, W) or 3D (H, W, 3) numpy array.")

    # Normalize kernel to 0-255 range
    kernel_min, kernel_max = np.min(kernel), np.max(kernel)
    kernel_normalized = ((kernel - kernel_min) / (kernel_max - kernel_min) * 255).astype(np.uint8)

    # Display the kernel
    plt.imshow(kernel_normalized)
    plt.axis('off')
    plt.colorbar(label='Pixel value')
    plt.title('Convolutional Kernel Visualization')
    plt.show()


if __name__ == '__main__':
    kernel = create_custom_kernel()
    show_kernel(kernel)

    image_path = ''
    process_layers(image_path)
