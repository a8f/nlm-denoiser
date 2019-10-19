#!/usr/bin/python3

from cv2 import imwrite

from denoiser import Denoiser

NOISY_IMAGE_FILENAME = "noisy.png"  # Input image
OUTPUT_IMAGE_FILENAME = "out.png"  # Output image

if __name__ == '__main__':
    print("Start")
    denoiser = Denoiser(NOISY_IMAGE_FILENAME)
    result = denoiser.denoise()
    if not imwrite(OUTPUT_IMAGE_FILENAME, result):
        print('Error writing image')
