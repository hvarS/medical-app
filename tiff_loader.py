
import os, psutil
import copy
import tifffile
from tifffile import TiffFile
import matplotlib.pyplot as plt

# process = psutil.Process(os.getpid())

# print(process.memory_info()) # memory utilised by process before loading

# everything in pixels
# memmap maps the memory-mapped-file region to the location in the hard disk rather than copying it to its memory map so that the process can access data directly from there.
# port = 1 if you want to copy the region to the process's memory map. 0 otherwise. (keep port = 1 if you want to do other stuff with this portion and do not want to load it again and again from the disk)
def tiff_parser(filename, topx, topy, height, width, port):
    tif = TiffFile(filename)
    image = tifffile.memmap(filename)
    cropped_image = image[topx : topx + height, topy : topy + height]
    # plt.imshow(cropped_image)
    # plt.show()
    return copy.deepcopy(cropped_image) if port else cropped_image

# cp = tiff_parser('sample_images_for_calibration/sample.tiff', 0, 0, 1000, 1000, 0)

# print(process.memory_info()) # memory utilised by process after loading
