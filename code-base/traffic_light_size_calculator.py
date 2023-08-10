from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

crop_x = 60
crop_up = 50
crop_down = 30

def extract_tl(image ,y,x):

    return image[  x - crop_x:x + crop_x,y - crop_up:y + crop_down]

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

    return clustered_image.astype(float)

def count_pixel_left_right(image, x, y, side):
    color = tuple(image[y, x])

    x_temp = x
    count = 0
    while x_temp>=0 and x_temp < image.shape[1] and tuple(image[y, x_temp]) == color:
        count += 1

        x_temp += side

    return count


def count_pixel_up_down(image, x, y, side):
    color = tuple(image[y, x])

    y_temp = y
    count = 0
    while y_temp >=0 and y_temp < image.shape[0] and tuple(image[y_temp, x]) == color:
        count += 1
        y_temp += side

    return count


def find_center_of_clustered__circle(image, x, y):
    count_pixel_l = count_pixel_left_right(image, x, y, -1)
    count_pixel_r = count_pixel_left_right(image, x, y, 1)
    print(count_pixel_r)
    center_x = x - count_pixel_l + (count_pixel_r +
                                    count_pixel_l) // 2

    count_pixel_u = count_pixel_up_down(image, x, y, -1)
    count_pixel_d = count_pixel_up_down(image, x, y, 1)

    center_y = y - count_pixel_u + (count_pixel_u +
                                    count_pixel_d) // 2

    image[center_y, center_x] = [0, 0, 255]

    return center_y, center_x, count_pixel_left_right(image , center_x, center_y, -1 )

def calc_tl_size(img, y, x):
    # get the tl from big image
    tl_img = extract_tl(img, x, y)

    plt.imshow(tl_img)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

    cluster_tl = cluster_image_colors(tl_img, 3)
    # Display the image using imshow()
    plt.imshow(cluster_tl)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

    # get the circle data by the index in the crop image
    x, y, radius = find_center_of_clustered__circle(cluster_tl, crop_up ,crop_x)

    cluster_tl[ x,y] = [1, 0, 0]

    # Display the image using imshow()
    plt.imshow(cluster_tl)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

    return tl_y - crop_up + s_y,tl_x-crop_x + s_x , radius

if __name__ == '__main__':

    image = mpimg.imread(r'C:\Users\dov31\Desktop\bootcamp\mobileye\mobileye-mobileye-group5\data\fullImages\aachen_000054_000019_leftImg8bit.png')
    tl_x = 804
    tl_y =  379
    s_x, s_y ,_ = calc_tl_size(image, tl_y, tl_x)
    image[  s_x, s_y]= [1,0,0]
    # Display the image using imshow()
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

