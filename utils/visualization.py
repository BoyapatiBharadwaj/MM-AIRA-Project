import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_image(image, title="Image"):
    plt.figure(figsize=(6,6))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def highlight_text(tokens, attributions):
    import matplotlib
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Reds')
    norm = matplotlib.colors.Normalize(vmin=attributions.min(), vmax=attributions.max())
    highlighted = ""
    for tok, attr in zip(tokens, attributions):
        color = matplotlib.colors.rgb2hex(cmap(norm(attr))[:3])
        highlighted += f"<span style='background-color:{color}'>{tok}</span> "
    return highlighted
