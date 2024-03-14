import torch
import numpy as np
from PIL import Image
from torch.nn.modules.utils import _pair

def block_scramble(image, block_size, random_shuffle=True, random_rotate=True, shuffle_rng=None, rotation_rng=None,
                   rotation_angles=[0,90,180,270]):
    """
    Scrambles a PIL image by dividing it into blocks of size MxN, randomly shuffling and rotating them.
    
    Updated from block_scramble adding separate seed for shuffle and rotation so you could look at how the
    two manipulations "combine" to give an overall effect (e.g., shuffle only, rotation only, or the exact
    same random shuffle and random rotation combined).
    
    :param image: PIL Image to be scrambled.
    :param block_size: Tuple (M, N) indicating the size of each block.
    :param rng: Optional numpy Random Number Generator or seed for reproducibility.
    :return: Scrambled and rotated PIL Image.
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Ensure the RNG is a numpy Random Number Generator
    if shuffle_rng is None:
        shuffle_rng = np.random.default_rng()
    elif isinstance(shuffle_rng, int):
        shuffle_rng = np.random.default_rng(shuffle_rng)
    
    if rotation_rng is None:
        rotation_rng = np.random.default_rng()
    elif isinstance(shuffle_rng, int):
        rotation_rng = np.random.default_rng(rotation_rng)
        
    M, N = block_size
    height, width, _ = img_array.shape

    # Calculate the number of blocks in each dimension
    num_blocks_vertical = height // M
    num_blocks_horizontal = width // N

    # Check if the image size is divisible by the block size
    if height % M != 0 or width % N != 0:
        raise ValueError("Image dimensions must be divisible by block size")

    # Create a list of blocks and rotate them randomly
    blocks = []
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            # Extract the block
            block = img_array[i*M:(i+1)*M, j*N:(j+1)*N, :]
            # Randomly rotate the block
            if random_rotate:
                rotation_angle = rotation_rng.choice(rotation_angles)
                rotated_block = Image.fromarray(block).rotate(rotation_angle)
                block = np.array(rotated_block)
            blocks.append(block)

    # Shuffle the blocks
    if random_shuffle:
        shuffle_rng.shuffle(blocks)

    # Reconstruct the scrambled and rotated image
    scrambled_img_array = np.vstack([np.hstack(blocks[i*num_blocks_horizontal:(i+1)*num_blocks_horizontal])
                                     for i in range(num_blocks_vertical)])

    # Convert numpy array back to PIL Image
    scrambled_image = Image.fromarray(scrambled_img_array)

    return scrambled_image


def grid_mask(image_size, block_dim, line_width=2, grid_color=[0, 0, 0], background_color=[1, 1, 1]):
    """
    Creates a mask with vertical and horizontal grid lines.
    
    :param image_size: Tuple (H, W) indicating the size of the image.
    :param block_dim: Integer N indicating size of each block (NxN).
    :param line_width: Integer indicating the width of the grid lines, must be a multiple of 2.
    :param grid_color: List [R, G, B] for the color of the grid lines.
    :param background_color: List [R, G, B] for the background color.
    :return: Numpy array representing the mask.
    """
    H, W = _pair(image_size)
    mask = np.zeros((H, W, 3), dtype=np.uint8)

    # Validate line_width
    if line_width % 2 != 0:
        raise ValueError("line_width must be a multiple of 2")

    # Half the line width to apply to both sides of the boundary
    half_lw = line_width // 2

    # Fill the background
    mask[:] = background_color
    
    # Set grid lines
    for i in range(0, H+1, H//block_dim):
        mask[max(i - half_lw, 0):min(i + half_lw, H), :] = grid_color
    for j in range(0, W+1, W//block_dim):
        mask[:, max(j - half_lw, 0):min(j + half_lw, W)] = grid_color
    
    return torch.tensor(mask).permute(2,0,1).float()