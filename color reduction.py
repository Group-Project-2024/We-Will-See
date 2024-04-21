import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation

original_image = cv2.imread("../coloring-by-numbers/images/love.jpeg")
img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# tsza gdzieś tu dorobić zmniejszanie obrazka bo dla takich dużcyh 4000x3000 długo mieli długo z 2,5 min

# noiseless_img = cv2.fastNlMeansDenoisingColored(original_image,None,30,30,21,41)

# Blurring
# blurred_img = cv2.GaussianBlur(original_image,(21,21),0)
blurred_img = cv2.medianBlur(original_image, 15)

# Denoising
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(blurred_img, kernel, iterations=2)
output = cv2.dilate(erosion, kernel, iterations=2)

attempts = 10

vectorized = output.reshape((-1, 3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 25  # (default = 17)
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape(img.shape)

# output = cv2.GaussianBlur(result_image, (3, 3), 0)
#
# vectorized = output.reshape((-1, 3))
# vectorized = np.float32(vectorized)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 17  # (default = 17)
# ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
# center = np.uint8(center)
# res = center[label.flatten()]
# result_image = res.reshape(img.shape)

# noiseless_image_colored = cv2.fastNlMeansDenoisingColored(result_image,None,20,20,7,21)

# Borders
image_shape = result_image.shape
borders = np.zeros((image_shape[0], image_shape[1]))

# Vertical
for i in range(image_shape[0]):
    for j in range(image_shape[1] - 1):
        if not np.array_equal(result_image[i][j], result_image[i][j + 1]):
            borders[i][j] = 1

# Horizontal
for i in range(image_shape[0] - 1):
    for j in range(image_shape[1]):
        if not np.array_equal(result_image[i][j], result_image[i + 1][j]):
            borders[i][j] = 1


result = cv2.imwrite('../coloring-by-numbers/images/love_median.jpg', result_image)

# # Create a kernel for erosion
# kernel_size = 3  # This size can be adjusted to control the amount of erosion
# kernel = np.ones((kernel_size, kernel_size), np.uint8)
#
# # Perform erosion
# eroded_matrix = cv2.erode(borders, kernel, iterations=1)  # Increase iterations for more erosion
# # Perform dilation to restore some size if needed
# thick_borders = cv2.dilate(eroded_matrix, kernel, iterations=1)
#borders = borders - thick_borders


# # Using skimage
# skeleton = skeletonize(borders)
#
# # Perform erosion
# eroded_matrix = binary_erosion(skeleton, structure=np.ones((3,3)))
#
# # Perform dilation to restore some size
# dilated_matrix = binary_dilation(eroded_matrix, structure=np.ones((3,3)))

# # Scipy Image
# # Calculate the distance transform
# dist_transform = distance_transform_edt(skeleton)
#
# # Threshold the distance transform to remove thick borders
# thin_matrix = dist_transform > 0.99  # Adjust the threshold as needed


# plt.figure(figsize=(32, 18))
# plt.imshow(dilated_matrix)
# plt.show()
