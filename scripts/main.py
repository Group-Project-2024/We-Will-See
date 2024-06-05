from functions import *
import matplotlib.pyplot as plt

path = "../images/image.jpg"
img = load_image(path)
out = segment_image(img, n_segments=5000, squared=30)

result_image = perform_kmeans_clustering(out, k=17, blur=True, blur_effect=21, save=True)
image_colors = return_colors(result_image)

borders_with_numbers = add_numbers_to_borders(result_image, image_colors)

plt.figure(figsize=(18, 32))
plt.imshow(borders_with_numbers)
plt.tick_params(left=False, right=False, labelleft=False,
                labelbottom=False, bottom=False)
plt.savefig('borders.pdf', bbox_inches='tight', pad_inches=0.02)

closest_paint_colors = find_closest_colors(image_colors)
image_paints = return_paint_names(closest_paint_colors)

closest_paint_colors = find_closest_colors(image_colors)

n = len(image_colors)

cmap_original = np.reshape(np.array(image_colors), (1, n, 3))
cmap_paints = np.reshape(np.array(closest_paint_colors), (1, n, 3))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 2), facecolor='white', gridspec_kw={'hspace': 0}, layout="tight")

ax1.imshow(cmap_original)
ax2.imshow(cmap_paints)

ax1.axis('off')
ax2.axis('off')

fig, axs = plt.subplot_mosaic("A;B", gridspec_kw={'height_ratios': [10, 0.3]}, figsize=(20, 37))

fig.tight_layout()
xticks = (np.arange(0, cmap_paints.shape[1] + 1, 1))
axs["A"].imshow(borders_with_numbers, cmap='Greys')
axs["A"].set_xticks([])
axs["A"].set_yticks([])

plt.xticks(range(0, cmap_paints.shape[1]))
axs["B"].imshow(cmap_paints)
for i, xpos in enumerate(axs["B"].get_xticks()):
    axs["B"].text(xpos, -.8, xticks[i],
                  size=30, ha='center')

    axs["B"].text(xpos, -1.25, image_paints[i],
                  size=30, ha='center')
axs["B"].axis('off')

plt.savefig('painting_sheet.pdf')
