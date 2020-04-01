import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL.Image import Image
from typing import Optional


def direct_display_image(image: Image, title: Optional[str] = None, titlesize=12):
    dpi = float(mpl.rcParams['figure.dpi'])

    w, h = image.size
    size = w / dpi, h / dpi

    fig = plt.figure(figsize=size)

    if title:
        title_y = 1 + 0.3 / size[1]
        fig.suptitle(title, fontsize=titlesize, y=title_y, weight='bold')
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')
    ax.imshow(image)
    plt.show()
