import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from anti_celiac.unet.data_generator import DataGenerator
from anti_celiac.unet.model import attention_unet_resnet50, attention_unet_refined
from anti_celiac.unet.metrics import *
import sys
import time

import cv2
from PIL import Image, ImageEnhance
from anti_celiac.unet.utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tqdm import tqdm
from anti_celiac.pyinstaller_relative import resource_path

import labelme 
import base64
import cv2
import json 
import yaml

CFG_FILE = resource_path('anti_celiac/config/unet_config.yaml')
CKPT = resource_path('..\..\weights\weights16.h5')
THRESHOLD = 0.5
TEMP_JSON_SAVE_LOC = resource_path(os.path.join(os.getcwd(),'temp_saver','tissue_annotations'))
DISPLAY_TEMP_SAVE = resource_path(os.path.join(os.getcwd(),'temp_saver','display'))
MASKS_TEMP_SAVE = resource_path(os.path.join(os.getcwd(),'temp_saver','masks'))

cfg = yaml.full_load(open(CFG_FILE,'r'))
num_classes = len(cfg["target_classes"])

if not os.path.exists(DISPLAY_TEMP_SAVE):
    os.makedirs(DISPLAY_TEMP_SAVE)

if not os.path.exists(TEMP_JSON_SAVE_LOC):
    os.makedirs(TEMP_JSON_SAVE_LOC)


if not os.path.exists(MASKS_TEMP_SAVE):
    os.makedirs(MASKS_TEMP_SAVE)


def initialise_json(image_path, image_size):
    json_saver = {}
    json_saver["imageHeight"] = image_size[1]
    json_saver["imageWidth"] = image_size[0]
    json_saver["version"] =  "5.1.1"
    json_saver["flags"] = {}
    json_saver["shapes"] = []
    data = labelme.LabelFile.load_image_file(image_path)
    image_data = base64.b64encode(data).decode('utf-8')
    json_saver["imageData"] = image_data
    json_saver["imagePath"] = image_path
    return json_saver


def get_contours(predicted_mask):
    contour_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
    for x in range(predicted_mask.shape[0]):
        for y in range(predicted_mask.shape[1]):
            contour_mask[x][y] = predicted_mask[x][y]
    
    ret, thresh = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def draw_shapes(clss, contours, scale_factors):
    shapes = []
    x_scale_factor, y_scale_factor = scale_factors
    for hull in contours:
        shape = {}
        shape['label'] = clss
        shape['points'] = []
        CLSS_CNT[clss] += 1

        hull = np.reshape(hull,(hull.shape[0],2))
        # hull = hull[::2,:] # take every 2nd point only
        for point in hull:
            add = [y_scale_factor*int(point[1]),x_scale_factor*int(point[0])]
            shape["points"].append(add)

        shape["group_id"] = CLSS_CNT[clss],
        shape["shape_type"] =  "polygon"
        shape["flags"] =  {}
        shapes.append(shape)
    return shapes



CLSS_CNT = {}


def magic(img_path):
    img_name = img_path.split('/')[-1].split('.')[0]
    target_classes = ["Crypts", "Villi", "Epithelium", "Brunner's Gland"]
    
    for clss in target_classes:
        CLSS_CNT[clss] = 0

    use_edges = False
    num_classes = len(target_classes) + int(use_edges)
    test_image_size = (320,256)
    display_image_size = (1280, 1024)

    x_scale_factor, y_scale_factor = display_image_size[0]/test_image_size[0], display_image_size[1]/test_image_size[1]

    _, pred_model, _, _, _ = attention_unet_refined(test_image_size,  
                                                    num_classes, 
                                                    multiplier=10, 
                                                    freeze_encoder=True, 
                                                    freeze_decoder=True, 
                                                    use_constraints = False, 
                                                    dropout_rate=0.0)
    pred_model.load_weights(CKPT, by_name=True)
    predictor = pred_model.predict_on_batch
    
   
    thresh_arg = float(THRESHOLD)

    if thresh_arg >= 0.:
        val = thresh_arg
        param_grid = [np.array([val] * num_classes).reshape((1, 1, -1))]
    else:
        param_grid = [np.array([val] * num_classes).reshape((1, 1, -1)) for val in np.linspace(0.0, 1.0, 101)]

        
    # image_path = os.path.join(test_folder, 'Images', test_id + '.jpg')
    img = Image.open(img_path).convert('RGB')
    w_img, h_img = img.size
    # w, h = img.size
    w, h = test_image_size
    if w_img >= 2 * w:
        w, h = 2 * w, 2 * h
    img = img.resize((w, h))
    # Preprocessing images
    display_img_resized = img.resize(display_image_size)
    display_image_save_path = resource_path(os.path.join(DISPLAY_TEMP_SAVE, img_name+'.png'))
    display_img_resized.save(display_image_save_path)

    img_resized = img.resize(test_image_size)
    
    ## Temporary Image Path
    
    """ Initialise Json after reducing the image size """
    json_saver = initialise_json(display_image_save_path, display_img_resized.size)

    img = np.array(img_resized)
    # Convert (H, W, C) to (W. H, C)
    img = np.transpose(img, (1, 0, 2))
    # img = np.clip(img - np.median(img)+127, 0, 255)
    img = img.astype(np.float32)
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img/255.0
    
    inputs = np.expand_dims(np.array(img),0)


    """Processing the input"""
    raw_image = np.transpose(inputs[0], (1, 0, 2))
    raw_image = (raw_image*255.0).astype(np.uint8)
    raw_image = Image.fromarray(raw_image)
    pink_mass = raw_image.copy()
    enhancer = ImageEnhance.Contrast(pink_mass)
    pink_mass = enhancer.enhance(4.0)
    pink_mass = np.array(pink_mass)
    pink_mass = 255 - ((pink_mass[:, :, 0] > 150) * (pink_mass[:, :, 1] > 150) * (pink_mass[:, :, 2] > 150)) * 255
    pink_mass = Image.fromarray(pink_mass.astype(np.uint8))


    """Making predictions"""
    inputs = tf.convert_to_tensor(inputs)
    result_mask = predictor(inputs)
    
    all_class_shapes = []

    for seg_threshold in (param_grid):
        # seg_threshold = np.array(list(seg_threshold.values()))
        common_mask_dir = resource_path(os.path.join(MASKS_TEMP_SAVE,img_name))
        if not os.path.exists(common_mask_dir):
            os.makedirs(common_mask_dir)
        """Processing the predictions for segmentation"""
        result_mask_thresh = (result_mask[0] > seg_threshold) * 1.
        for i,clss in enumerate(target_classes):

            new_mask_dir = os.path.join(common_mask_dir,clss)
            if not os.path.exists(new_mask_dir):
                os.makedirs(new_mask_dir)
            

            predicted_mask = result_mask_thresh[...,i] * 255
            ret, thresh = cv2.threshold(predicted_mask, 127, 255, cv2.THRESH_BINARY) # Thresh is the binary mask
            
            thresh = cv2.resize(thresh,display_image_size)

            mask_file_path = resource_path(os.path.join(new_mask_dir,img_name+'.png'))
            cv2.imwrite(mask_file_path,thresh)

            contours = get_contours(predicted_mask)
            shapes = draw_shapes(clss,contours,(x_scale_factor,y_scale_factor))
            all_class_shapes.extend(shapes)
    
        result_mask_img = output2image(result_mask_thresh)


    json_saver["shapes"] = all_class_shapes
    json_object = json.dumps(json_saver)  


    """Combined visualization"""
    # if thresh_arg >= 0.:
    #     rmw, rmh = result_mask_img.size
    #     divider = Image.new('RGB', (10, rmh), (255, 255, 255))
    #     combined_img = Image.new('RGB', (rmw*3 + 20, rmh))
    #     combined_img.paste(raw_image, (0, 0))
    #     combined_img.paste(divider, (rmw, 0))
    #     combined_img.paste(result_mask_img, (rmw + 10, 0))
    #     combined_img.save(os.path.join('./Results',f'Generated_{display_image_size[0]}_{display_image_size[1]}.png'))

    """Save the Json Dumped as a string"""
    json_file_location = resource_path(os.path.join(TEMP_JSON_SAVE_LOC,img_name+'.json'))
    with open(json_file_location,'w') as f:
        f.write(json_object)

    return json_file_location, display_image_save_path
    



