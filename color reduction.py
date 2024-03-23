import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("../coloring-by-numbers/images/jakub.jpg")
img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# tsza gdzieś tu dorobić zmniejszanie obrazka bo dla takich dużcyh 4000x3000 długo mieli długo z 2,5 min

# noiseless_image = cv2.fastNlMeansDenoisingColored(original_image,None,30,30,21,41)

# noiseless_image = cv2.blur(original_image,(10,10)) #do wyjebania
# noiseless_image = cv2.GaussianBlur(original_image,(21,21),0) # gaus też spoko
noiseless_img = cv2.medianBlur(original_image, 15)

# tu dodałam tylko
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(noiseless_img, kernel, iterations=2)
output = cv2.dilate(erosion, kernel, iterations=2)

# noiseless_image = cv2.blur(original_image,(10,10)) # tu przetestować tego czwartego

vectorized = output.reshape((-1, 3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 17  # (default = 17)
attempts = 10
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape((img.shape))

# Granice
image_shape = result_image.shape
borders = np.zeros((image_shape[0], image_shape[1]))

# tu dodajemy granice bardziej pionowe
for i in range(image_shape[0]):
    for j in range(image_shape[1] - 1):
        if not np.array_equal(result_image[i][j], result_image[i][j + 1]):
            borders[i][j] = 1

# a tu bardziej poziome
for i in range(image_shape[0] - 1):
    for j in range(image_shape[1]):
        if not np.array_equal(result_image[i][j], result_image[i + 1][j]):
            borders[i][j] = 1

# noiseless_image_colored = cv2.fastNlMeansDenoisingColored(result_image,None,20,20,7,21)

result = cv2.imwrite('../coloring-by-numbers/images/result.jpg', result_image)
