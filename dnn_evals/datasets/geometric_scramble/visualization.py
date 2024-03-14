import matplotlib.pyplot as plt
from math import ceil

__all__ = ['imshow_scrambled']

def imshow_scrambled(imgs, titles, num_cols=4, figsize=None):
    """
    Plots images in a grid with titles from a list of dictionaries.

    Args:
        imgs (list): List of PIL Image objects.
        titles (list): List of dictionaries, one for each image.
        num_cols (int): Number of columns in the image grid.
        figsize (tuple, optional): Figure size. If None, default is used.
    """
    assert len(imgs) == len(titles), "The length of images and titles must match"
    
    num_rows = ceil(len(imgs) / num_cols)  # Calculate the number of rows needed
    if figsize is None:
        figsize = (num_cols * 4, num_rows * 4)  # Default size, can be adjusted
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the grid to easily loop over it
    
    for i, (img, title_dict) in enumerate(zip(imgs, titles)):
        # Convert title dictionary to string
        title_str = ', '.join(f'{key}: {value}' for key, value in title_dict.items() 
                              if value != 0 and not isinstance(value, str))
        axes[i].imshow(img)
        axes[i].set_title(title_str, fontsize=10)
        axes[i].axis('off')  # Hide axes ticks

    # Hide unused subplots if any
    for ax in axes[len(imgs):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()