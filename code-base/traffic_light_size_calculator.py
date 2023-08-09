from sklearn.cluster import KMeans

crop_x = 60
crop_up = 50
crop_down = 30
y = (400, 1133)


def extract_tl(image, x, y):
    return image[y - crop_up:y + crop_down, x - crop_x:x + crop_x]


def cluster_image_colors(image, num_clusters):
    """
    Cluster colors within a single image using k-means algorithm.

    Args:
    image (numpy array): Image data as a numpy array.
    num_clusters (int): Number of clusters to create.

    Returns:
    numpy array: Image with clustered colors.
    """

    # Flatten the image into a list of pixels
    image_pixels = image.reshape((-1, 3))

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(image_pixels)

    # Replace each pixel with its cluster center
    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape the clustered pixels back into an image
    clustered_image = clustered_pixels.reshape(image.shape)

    return clustered_image.astype(int)
