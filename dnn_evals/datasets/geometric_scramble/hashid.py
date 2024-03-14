import numpy as np
import hashlib
from PIL import Image

__all__ = ['hash_pil_image']

def hash_pil_image(img, hash_func='md5'):
    """Compute hash for a PIL Image."""
    np_arr = np.array(img)  # Convert PIL Image to numpy array
    if img.mode == 'RGBA':  # If the image has an alpha channel, consider it as well
        np_arr = np_arr.flatten()  # Flatten the array for hashing
    else:
        np_arr = np_arr[:,:,0]  # Use only one channel for grayscale or RGB images for consistency
    
    # Choose the hashing function
    if hash_func == 'md5':
        hasher = hashlib.md5()
    elif hash_func == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError("Unsupported hash function")

    # Update the hasher with the array bytes
    hasher.update(np_arr.tobytes())
    return hasher.hexdigest()