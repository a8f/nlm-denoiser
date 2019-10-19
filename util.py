import numpy as np
from cv2 import imread


def load_image(filename: str) -> np.ndarray:
    """
    Load image at filename, raising FileNotFound if the loaded file's image data is not accessible
    :param filename: the file to load the image from
    :return: loaded image as ndarray
    :raises: FileNotFoundError if image cannot be read or image is read with no data
    """
    image = imread(filename, -1)
    if image is None or image.data is None:
        raise FileNotFoundError(filename)
    return image


def generate_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Return a patch_size*image.shape ndarray of patches in image with NaN for pixels outside the original image
    :param image: the image to get patches from
    :param patch_size: patch side length
    :return: patch_size*image.shape ndarray of patches in the image with NaN padding
    """
    # Pad image with NaN for patch_size//2 pixels on all sides
    padded_shape = image.shape[0] + patch_size - 1, image.shape[1] + patch_size - 1, image.shape[2]
    padded_im = np.empty(padded_shape)
    padded_im[:] = np.nan
    half_patch = patch_size // 2
    padded_im[half_patch:(image.shape[0] + half_patch), half_patch:(image.shape[1] + half_patch), :] = image
    # Array of patches for each pixel
    patches = np.empty((image.shape[0], image.shape[1], image.shape[2], patch_size ** 2))
    patches[:] = np.nan
    for i in range(patch_size):
        for j in range(patch_size):
            patches[:, :, :, i * patch_size + j] = padded_im[i:(i + image.shape[0]), j:(j + image.shape[1]), :]
    return patches


def coords_matrix(shape: list) -> np.ndarray:
    """
    Return a shape.shape[0]*shape.shape[1]*2 array f such that f[x, y] = [y,x]
    :param shape: shape of the input image
    :return: shape.shape[0]*shape.shape[1]*2 array f such that f[x, y] = [y,x]
    """
    range_x = np.arange(0, shape[1])
    range_y = np.arange(0, shape[0])
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)
    return np.dstack((axis_y, axis_x))
