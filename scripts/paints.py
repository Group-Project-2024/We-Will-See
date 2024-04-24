import numpy as np
import pandas as pd
import cv2
import os


def get_color(image):
    # Convert image data type to float and scale down (if required by your image data type handling)
    data = image.astype(np.float32) / 255.0
    # Count unique colors in the image excluding black [0,0,0] and white [1,1,1] (after scaling)
    colors, count = np.unique(data.reshape(-1, 3), axis=0, return_counts=True)
    # Filter out black and white
    valid_colors = colors[(~np.all(colors == [0, 0, 0], axis=1)) & (~np.all(colors == [1, 1, 1], axis=1))]
    # Select the most frequent valid color
    if valid_colors.size > 0:
        return (valid_colors[np.argmax(
            count[(~np.all(colors == [0, 0, 0], axis=1)) & (~np.all(colors == [1, 1, 1], axis=1))])] * 255).astype(int)
    else:
        return None


colors_path = os.path.join(os.path.dirname(os.getcwd()), "colors")
color_files = os.listdir(colors_path)
color_names = [os.path.splitext(file)[0].upper() for file in color_files]

rgb_colors = []

for filename in color_files:
    file_path = os.path.join(colors_path, filename)
    if os.path.exists(file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dominant_color = get_color(img)
            if dominant_color is not None:
                rgb_colors.append(dominant_color.tolist())

df = pd.DataFrame({
    "color_name": color_names,
    "rgb": rgb_colors
})

df.to_csv("paints.csv", index=False)
