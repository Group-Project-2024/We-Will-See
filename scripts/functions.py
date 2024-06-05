import numpy as np
import cv2
import skimage
import os
import ast
import pandas as pd


# def create_borders(image):
#     image_shape = image.shape
#     borders = np.zeros((image_shape[0], image_shape[1]))
#
#     # Vertical
#     for i in range(image_shape[0]):
#         for j in range(image_shape[1] - 1):
#             if not np.array_equal(image[i][j], image[i][j + 1]):
#                 borders[i][j] = 1
#
#     # Horizontal
#     for i in range(image_shape[0] - 1):
#         for j in range(image_shape[1]):
#             if not np.array_equal(image[i][j], image[i + 1][j]):
#                 borders[i][j] = 1
#
#     return borders


def add_numbers_to_borders(kmeans_image, colors):
    # Prepare the image for drawing (convert single channel mask to BGR if necessary)
    finished = False
    borders_with_numbers = np.ones_like(kmeans_image) * 255

    for i, color in enumerate(colors):
        # Create a mask for the current color
        mask = cv2.inRange(kmeans_image, color, color)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the distance transform of the binary image within the contour
            mask_contour = np.zeros_like(mask)
            cv2.drawContours(borders_with_numbers, [contour], 0, (0, 0, 0), thickness=1)
            cv2.drawContours(mask_contour, [contour], 0, 255, thickness=cv2.FILLED)
            dist_transform = cv2.distanceTransform(mask_contour, cv2.DIST_L2, 3)

            # Find the indices of the maximum value in the distance transform
            max_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)

            # Center of the largest inscribed circle
            cx, cy = max_idx[::-1]  # Reverse the order for (x, y)

            # Adjust text size based on the radius
            max_val = dist_transform[max_idx]
            font_scale = min(max_val / 40, 1)  # Adjust font size based on distance, but limit to a maximum

            # Define text and font
            text = f"{i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (0, 0, 0)  # white text

            # Place text at the circle's center
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            text_x = max(10, min(cx - text_size[0] // 2, borders_with_numbers.shape[1] - text_size[0]))
            text_y = max(text_size[1] + 10, min(cy + text_size[1] // 2, borders_with_numbers.shape[0]))
            saved_x = text_x
            saved_y = text_y
            # Boundary checks for text_x and text_y
            # if text_x <= 50 or text_x >= (mask.shape[1] - 50) or text_y <= 50 or text_y >= (mask.shape[0] - 50):
            cv2.putText(borders_with_numbers, text, (text_x, text_y), font, font_scale, font_color, 1, cv2.LINE_AA)
            # else:
            #     #try:
            #     #     while not (mask[text_y - 1][text_x - 1] == 255 and
            #     #                mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #     #                mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #     #                mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #     #         text_x += 5
            #     #         text_y += 5
            #     #
            #     # except IndexError:
            #     #     text_x = saved_x
            #     #     text_y = saved_y
            #     #
            #     # try:
            #     #     while not (mask[text_y - 1][text_x - 1] == 255 and
            #     #                mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #     #                mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #     #                mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #     #         text_x += 5
            #     #         text_y -= 5
            #     #
            #     # except IndexError:
            #     #     text_x = saved_x
            #     #     text_y = saved_y
            #     #
            #     # try:
            #     #     while not (mask[text_y - 1][text_x - 1] == 255 and
            #     #                mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #     #                mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #     #                mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #     #         text_x -= 5
            #     #         text_y += 5
            #     #
            #     # except IndexError:
            #     #     text_x = saved_x
            #     #     text_y = saved_y
            #
            #     try:
            #         while not (mask[text_y - 1][text_x - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #                    mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #             text_x -= 5
            #             text_y -= 5
            #
            #     except IndexError:
            #         text_x = saved_x
            #         text_y = saved_y
            #
            #     try:
            #         while not (mask[text_y - 1][text_x - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #                    mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #             text_x += 1
            #
            #     except IndexError:
            #         text_x = saved_x
            #         text_y = saved_y
            #
            #     try:
            #         while not (mask[text_y - 1][text_x - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #                    mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #             text_x -= 1
            #
            #     except IndexError:
            #         text_x = saved_x
            #         text_y = saved_y
            #
            #     try:
            #         while not (mask[text_y - 1][text_x - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #                    mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #             text_y += 1
            #
            #     except IndexError:
            #         text_x = saved_x
            #         text_y = saved_y
            #
            #     try:
            #         while not (mask[text_y - 1][text_x - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x - 1] == 255 and
            #                    mask[text_y - 1][text_x + text_size[0] - 1 - 1] == 255 and
            #                    mask[text_y + text_size[1] - 1 - 1][text_x + text_size[0] - 1 - 1] == 255):
            #             text_y -= 1
            #
            #     except IndexError:
            #         text_x = saved_x
            #         text_y = saved_y
            #
            #     cv2.putText(borders_with_numbers, text, (text_x, text_y), font, font_scale, font_color, 1, cv2.LINE_AA)

    return borders_with_numbers


def load_image(path, target_size=1670):
    image = skimage.io.imread(path)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Determine the scale factor
    if height < width:
        scale_factor = target_size / height
    else:
        scale_factor = target_size / width

    # Compute the new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Resize the image
    resized_image = skimage.transform.resize(image, (new_height, new_width), anti_aliasing=True)

    if resized_image.dtype != np.uint8:
        resized_image = (resized_image * 255).astype(np.uint8)

    return resized_image


def perform_kmeans_clustering(image, k=17, blur=True, blur_effect=11, save=False,
                              path=os.path.join(os.path.dirname(__file__), 'morze.jpg')):
    if blur is True:
        image = cv2.medianBlur(image, blur_effect)

    vectorized = image.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(image.shape)

    if save is True:
        img = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, img)

    return result_image


def return_colors(image):
    # Reshape the image to a 2D array where each row represents a color (R, G, B)
    reshaped_image = image.reshape(-1, 3)

    # Find unique colors and their corresponding counts
    unique_colors, counts = np.unique(reshaped_image, axis=0, return_counts=True)

    # Sort the unique colors by frequency (descending order)
    # argsort returns the indices that would sort an array, and we use it on counts
    sorted_indices = np.argsort(-counts)
    sorted_colors = unique_colors[sorted_indices]

    return sorted_colors


def euclidean_distance(color1, color2):
    # Convert colors to numpy arrays of type float for accurate calculation
    return np.sqrt(np.sum((np.array(color1, dtype=float) - np.array(color2, dtype=float)) ** 2))


def find_closest_colors(image_colors):
    paints = pd.read_csv("paints.csv")
    paint_rgbs = np.array([ast.literal_eval(rgb) for rgb in paints['rgb']])

    available_paints = paint_rgbs.copy()
    closest_colors = []

    for image_color in image_colors:
        distances = [euclidean_distance(image_color, np.array(paint_color, dtype=float)) for paint_color in
                     available_paints]

        if distances:
            closest_color_index = np.argmin(distances)
            closest_color = available_paints[closest_color_index]
            available_paints = np.delete(available_paints, closest_color_index, axis=0)
            closest_colors.append(closest_color)
        else:
            closest_colors.append(None)

    return np.array(closest_colors)


def return_paint_names(matching_paints_rgbs):
    paints = pd.read_csv("paints.csv")
    paint_names = np.array(paints['paint_name'])
    paint_rgbs = np.array([ast.literal_eval(rgb) for rgb in paints['rgb']])

    return [paint_names[np.where((paint_rgbs == color).all(axis=1))][0] for color in matching_paints_rgbs]


def segment_image(image, n_segments=5000, squared=30):
    segments_labels = skimage.segmentation.slic(image, compactness=squared, n_segments=n_segments, start_label=0)
    segmented_image = skimage.color.label2rgb(segments_labels, image, kind='avg', bg_label=1)

    return segmented_image


def add_numbers_to_borders(kmeans_image, borders, colors):
    # Prepare the image for drawing (convert single channel mask to BGR if necessary)
    borders = skimage.morphology.binary_closing(borders)
    borders_with_numbers = borders * 255

    for i, color in enumerate(colors):
        # Create a mask for the current color
        mask = cv2.inRange(kmeans_image, color, color)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the distance transform of the binary image within the contour
            mask_contour = np.zeros_like(mask)
            cv2.drawContours(mask_contour, [contour], 0, 255, thickness=cv2.FILLED)
            dist_transform = cv2.distanceTransform(mask_contour, cv2.DIST_L2, 3)

            # Find the indices of the maximum value in the distance transform
            max_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)

            # Center of the largest inscribed circle
            cx, cy = max_idx[::-1]  # Reverse the order for (x, y)

            # Adjust text size based on the radius
            max_val = dist_transform[max_idx]
            font_scale = min(max_val / 40, 1)  # Adjust font size based on distance, but limit to a maximum

            # Define text and font
            text = f"{i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)  # white text

            # Place text at the circle's center
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            text_x = max(0, min(cx - text_size[0] // 2, borders_with_numbers.shape[1] - text_size[0]))
            text_y = max(text_size[1], min(cy + text_size[1] // 2, borders_with_numbers.shape[0]))
            cv2.putText(borders_with_numbers, text, (text_x, text_y), font, font_scale, font_color, 1, cv2.LINE_AA)

    return borders_with_numbers
