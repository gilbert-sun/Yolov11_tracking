

import numpy as np
import cv2
from argparse import ArgumentParser

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, default='/home/k900/Downloads/60885.0_20250410_161050')
    parser.add_argument('--result_folder', type=str, default="out_padded_img")
    parser.add_argument('--width', type=int, default=8320)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--data_type', type=str, default='uint8')
    args = parser.parse_args()
    #print(args)

    #Constants
    width = args.width
    dtype = args.data_type #np.uint8
    filename = args.video_folder #'/home/k900/Downloads/60885.0_20250410_161050'
    channel = args.channel
    fout = args.result_folder

    # Step 1: Read binary data
    with open(filename, 'rb') as f:
        raw_data = f.read()

    array = np.frombuffer(raw_data, dtype=dtype)

    # Step 2: Calculate padding
    remainder = array.size % width
    if remainder != 0:
        padding = width - remainder
        array = np.pad(array, (0, padding), mode='constant')
    else:
        padding = 0
    #
    # print("--------------< width> : <remainder> ==> ",width ,remainder)
    # print("--------------< padding> : <array> ==> ", padding, len(array))
    # Step 3: Reshape into 2D image
    height = array.size // width // channel
    image_2d = array.reshape((height, width, channel))

    # print("--------------< height> : <array> ==> ", height, len(image_2d))
    # print(f"Final shape: {image_2d.shape} (padded with {padding} zero bytes)")

    # Step 4: Save and display
    normalized_img = cv2.convertScaleAbs(image_2d, alpha=(255.0 / 65535.0))
    #cv2.imwrite('W_{}_H_{}.png'.format(width,height), image_2d)
    cv2.imwrite(fout+'_W_{}_H_{}_CH_{}.png'.format(width,height,channel), image_2d)
    #cv2.imwrite('output_padded_image.png', image_2d)
    cv2.imshow('Padded Image (8-bit)', image_2d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
