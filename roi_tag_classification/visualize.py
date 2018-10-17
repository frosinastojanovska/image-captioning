import random
import colorsys
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt


def draw_boxes_and_tags(image, boxes, tags, title='', file_path=None):
    _, ax = plt.subplots(1, figsize=(12, 12))
    n = boxes.shape[0]
    colors = random_colors(n)

    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    drawn_image = image.astype(np.uint32).copy()
    for i in range(n):
        color = colors[i]
        style = "solid"
        alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Captions
        if tags is not None:
            tags_current = tags[i]
            ax.text(x1, y1, tags_current, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

    ax.imshow(drawn_image.astype(np.uint8))
    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()


def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
