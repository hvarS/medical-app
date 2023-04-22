import numpy as np
import pandas as pd
import torch
import os
import cv2
import argparse
import sys
from caranet.utils.dataloader import get_loader, test_dataset, inference_dataset
import torch.nn.functional as F
import numpy as np
from caranet.CaraNet import caranet
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import labelme
import base64

save_imgs = True # if False, then just predict segmentation masks, otherwise predict masked image

'''
Input: Path to folder containing images
Output: Predicted Segmentation mask

Check:
Should work for both CPU and GPU
Post process the small image blobs
'''

class Inference_Villi:
    def __init__(self, model_path):
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model = self.load_model()

    def load_model(self):
        model = caranet()
        weights = torch.load(self.model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            if 'total_ops' not in k and 'total_params' not in k:
                name = k
                new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device=self.device)
        return model

    def remove_small_blob(self, res):
        im_gauss = cv2.GaussianBlur(res, (5, 5), 0)
        ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)
        thresh = thresh.astype(np.uint8)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_area = []
        #create a mask
        mask = np.ones(res.shape[:2], dtype="uint8") * 255
        # calculate area and filter into new array
        for con in contours:
            area = cv2.contourArea(con)
            if 500 < area < 10000:
                cv2.drawContours(mask, [con], -1, 0, -1)
                contours_area.append(con)
        process_image = cv2.bitwise_and(res, res, mask=mask)
        return process_image


    def get_predictions(self, image_path):
        #load single image
        inference_loader = inference_dataset(image_path)
        image = inference_loader.load_data()
        if self.device == torch.device('cuda:0'):
            image = image.cuda()
        res5,res4,res2,res1 = self.model(image)
        res = res5
        res = F.upsample(res, size= (640, 640), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res>0.3).astype('float')
        res = res * 255
        image = image.squeeze()
        image = image.permute(1,2,0)
        image = image.cpu().detach().numpy()
        image = (image - image.min())/(image.max() - image.min())
        process_image = self.remove_small_blob(res)
        return process_image



def infer(img_location):
    #checking the inference
    model_save_path = '../../weights/CaraNet-best_84.pth'
    temp_save_dir = os.path.join(os.getcwd(),'temp_saver','results','segmented_images')
    temp_resized_save_dir = os.path.join(os.getcwd(),'temp_saver','results','interpretable_resized')
    temp_json_save_dir = os.path.join(os.getcwd(),'temp_saver','results','annotations')
    
    if not os.path.exists(temp_save_dir):
        os.makedirs(temp_save_dir)
    
    if not os.path.exists(temp_json_save_dir):
        os.makedirs(temp_json_save_dir)

    if not os.path.exists(temp_resized_save_dir):
        os.makedirs(temp_resized_save_dir)

    # Jsonify Output File
    image_path = img_location

    img_name = img_location.split('/')[-1][:-4]
    


    Inference_villi = Inference_Villi(model_save_path)
    predicted_mask = Inference_villi.get_predictions(image_path)
    
    imageHeight = 640
    imageWidth = 640

    img = cv2.imread(image_path)
    img = cv2.resize(img,(imageHeight,imageWidth))
    temp_resized_image_path = os.path.join(temp_resized_save_dir,img_name+'.png')
    cv2.imwrite(temp_resized_image_path,img)

   
    ## -------- ##
    json_saver = {}
    json_saver["imageHeight"] = imageHeight
    json_saver["imageWidth"] = imageWidth
    json_saver["version"] =  "5.1.1"
    json_saver["flags"] = {}
    json_saver["shapes"] = []
    data = labelme.LabelFile.load_image_file(temp_resized_image_path)
    image_data = base64.b64encode(data).decode('utf-8')
    json_saver["imageData"] = image_data
    ## -------- ##
        

    
    predicted_mask = (predicted_mask[:, :] > 0.1) * 255
    #####
    contour_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
    for x in range(predicted_mask.shape[0]):
        for y in range(predicted_mask.shape[1]):
            contour_mask[x][y] = predicted_mask[x][y]
    
    ret, thresh = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #######
    
    predicted_mask = np.expand_dims(predicted_mask // 255, axis=-1)
    
    shapes = []

    for idx,hull in enumerate(contours):
        shape = {}
        shape['label'] = 'Inference'
        shape["line_color"] = [
            0,
            10,
            0,
            20
        ],
        shape["fill_color"] =  [
            0,
            20,
            0,
            40
        ]
        shape['points'] = []

        hull = np.reshape(hull,(hull.shape[0],2))
        hull = hull[::6,:] # take every 6th point only
        for point in hull:
            add = [int(point[0]),int(point[1])]
            shape["points"].append(add)
        
        mask_array = np.zeros((imageHeight,imageWidth),dtype=np.int32)
        cv2.fillPoly(mask_array, pts =[np.array(shape["points"])], color=(255,255,255))

        areaContour = 0
        for x in range(imageHeight):
            for y in range(imageWidth):
                if mask_array[x][y] == 255:
                    areaContour += 1
        areaLimit = imageHeight*imageWidth//1000

        if areaContour<areaLimit:
            continue

        shape["group_id"] = idx+1,
        shape["shape_type"] =  "polygon"
        shape["flags"] =  {}
        shapes.append(shape)

    img = img*predicted_mask
    cv2.imwrite(os.path.join(temp_save_dir,img_name+'.png'),img)    
    image_path =  os.path.join(temp_save_dir,img_location)
    json_saver["imagePath"] = image_path

    json_saver["shapes"] = shapes
    json_object = json.dumps(json_saver)     

    json_path = os.path.join(temp_json_save_dir,img_name+'.json')
    
    with open(json_path,'w') as f:
        f.write(json_object)
    
    f.close()

    img_path  = os.path.join(temp_save_dir,img_name+'.png')

    return json_path, img_path, temp_resized_image_path

