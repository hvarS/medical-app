import cv2
import os
from PIL import Image
from pyinstaller_relative import resource_path
save_dir = resource_path(os.path.join(os.getcwd(),'temp_saver','results','patched_results'))
interpretable_dir = resource_path(os.path.join(os.getcwd(),'temp_saver','results','interpretable_images'))

def create_patches(image_path):
    img = cv2.imread(image_path)

    y_ = img.shape[0]
    x_ = img.shape[1]

    x_scale = 1920 / x_
    y_scale = 1920 / y_

    img = cv2.resize(img , (1920,1920), interpolation = cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    ## Save Bigger Image 
    img_name = image_path.split('\\')[-1][:-4]
    patch_img_path_dir = resource_path(os.path.join(save_dir,img_name))

    if not os.path.exists(patch_img_path_dir):
        os.makedirs(patch_img_path_dir)
    spath = resource_path(os.path.join(save_dir, img_name+ '.png'))
    ipt_img_path = spath
    img.save(spath)

    ##                 ##
    from itertools import product

    w, h = img.size
    d = 640
    k = 0
    

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        spath = resource_path(os.path.join(patch_img_path_dir , img_name + '_{}.png'.format(k)))
        img.crop(box).save(spath)
        k += 1

    return ipt_img_path, patch_img_path_dir