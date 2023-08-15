import cv2
from PIL import Image
import numpy as np
import glob
import os
import fnmatch
import labelme
import base64
import json


CLASS_MAPPING = {
    0: 'Epithelial Nuclei',
    1: 'IEL'
}

MULTIPLIER = 640
CLASS_COUNTER = {}

def color(clss, img, x_off, y_off, start_x, start_y, height, width) -> dict:
    clr = [(0,0,255),(0,255,0)]
    if clss not in CLASS_COUNTER:
        CLASS_COUNTER[clss] = 1
    if clss == 0:
        shape = {}
        shape['label'] = CLASS_MAPPING[clss]
        shape["line_color"] = [
            0,
            128,
            0,
            32
        ],
        shape["fill_color"] =  [
            0,
            240,
            0,
            32,
        ]
        shape['points'] = []
    else:
        shape = {}
        shape['label'] = CLASS_MAPPING[clss]
        shape["line_color"] = [
            85,
            0,
            0,
            32
        ],
        shape["fill_color"] =  [
            170,
            0,
            0,
            64,
        ]
        shape['points'] = []

    shape["group_id"] = CLASS_COUNTER[clss],
    CLASS_COUNTER[clss] += 1
    
    shape["shape_type"] =  "polygon"
    shape["flags"] =  {}


    point = [y_off+start_y-height//2,x_off+start_x-width//2]
    shape["points"].append(point)
    for x in range(x_off+start_x-width//2,min(x_off+start_x+width//2,x_off+639)):
        img[x][y_off+start_y-height//2][:] = clr[clss]
    
    point = [y_off+start_y-height//2,x_off+start_x+width//2]
    shape["points"].append(point)
    for x in range(x_off+start_x-width//2,min(x_off+start_x+width//2,x_off+639)):
        img[x][min(y_off+start_y+height//2,y_off+639)][:] = clr[clss]
    
    point = [y_off+start_y+height//2,x_off+start_x+width//2]
    shape["points"].append(point)
    
    for y in range(y_off+start_y-height//2,min(y_off+start_y+height//2,y_off+639)):
        img[x_off+start_x-width//2][y][:] = clr[clss]
    
    point = [y_off+start_y+height//2,x_off+start_x-width//2]
    shape["points"].append(point)
    for y in range(y_off+start_y-height//2,min(y_off+start_y+height//2,y_off+639)):
        img[min(x_off+start_x+width//2,x_off+639)][y][:] = clr[clss]

    return shape


def color_rectangle(clss, img, start_x, start_y, height, width) -> dict:
    shape = {}
    shape['label'] = 'IR'
    shape["line_color"] = [
        0,
        128,
        0,
        32
    ],
    shape["fill_color"] =  [
        0,
        240,
        0,
        32,
    ]
    shape['points'] = []
    

    shape["group_id"] = 1,
    
    shape["shape_type"] =  "rectangle"
    shape["flags"] =  {}


    point = [start_x-width//2,start_y-height//2]
    shape["points"].append(point)
    
    point = [start_x+width//2,start_y+height//2]
    shape["points"].append(point)
    
    return shape


from pyinstaller_relative import resource_path

def get_json_from_labels_iel(image_file_path:str, labels_path):
    
    temp_save_dir = resource_path(os.path.join(os.getcwd(),'temp_saver','results','counting_annotations'))
    if not os.path.exists(temp_save_dir):
        os.makedirs(temp_save_dir)

    
    actual_height = 1920
    actual_width = 1920


    # Resize image and rewrite 
    img = cv2.imread(image_file_path)
    
    y_ = img.shape[0]
    x_ = img.shape[1]

    x_scale = 1920 / x_
    y_scale = 1920 / y_

    img = cv2.resize(img , (1920,1920), interpolation = cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    ## -------- ##
    ## Defining Basic Json Characteristics ##
    json_saver = {}
    json_saver["imageHeight"] = actual_height
    json_saver["imageWidth"] = actual_width
    json_saver["version"] =  "5.1.1"
    json_saver["flags"] = {}
    json_saver["shapes"] = []
    data = labelme.LabelFile.load_image_file(image_file_path)
    image_data = base64.b64encode(data).decode('utf-8')
    json_saver["imageData"] = image_data
    json_saver["imagePath"] = image_file_path
    ## -------- ##


    ## Get Grid to match the patch locations 
    w, h = actual_height, actual_width
    d = 640
    k = 0
    from itertools import product
        
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    grid = [(x,y) for x,y in grid]

    img = Image.fromarray(img)

    np_img = np.array(img)
    img_name = image_file_path.split("/")[-1].split('.')[0]

    label_files = []
    shapes = []

    for file in os.listdir(labels_path):
        if fnmatch.fnmatch(file, img_name+'*'):
            label_files.append(file)

    for label_file_path in label_files:


        label_file = open(os.path.join(labels_path,label_file_path),'r')
        idx = int(label_file_path.split('/')[-1].split('.')[0].split('_')[-1])

        for i,line in enumerate(label_file):
            clss, y, x, width, height , confidence_score = line.split(' ')        
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
            clss = int(clss)

            rel_x_location = round(MULTIPLIER*x)
            rel_y_location = round(MULTIPLIER*y)
            rel_width = round(MULTIPLIER*width)
            rel_height = round(MULTIPLIER*height)
            

                
            if float(confidence_score)>0.3:
                shape = color(clss, np_img, grid[idx][0], grid[idx][1],  rel_x_location, rel_y_location, rel_width, rel_height)
                shapes.append(shape)

    json_saver["shapes"] = shapes
    json_object = json.dumps(json_saver) 

    json_save_path = resource_path(os.path.join(temp_save_dir,img_name+'_iel.json'))
    with open(json_save_path,'w') as f:
        f.write(json_object)
    
    return json_save_path

def get_json_from_labels_ir(image_file_path:str, json_file_path, labels_path):

    temp_save_dir = resource_path(os.path.join(os.getcwd(),'temp_saver','ir_annotations'))
    if not os.path.exists(temp_save_dir):
        os.makedirs(temp_save_dir)

    img = cv2.imread(image_file_path)
    actual_height = img.shape[1]
    actual_width = img.shape[0]

    json_saver = json.load(open(json_file_path,'r'))
    
    y_ = img.shape[0]
    x_ = img.shape[1]


    np_img = np.array(img)
    img_name = image_file_path.split("/")[-1].split('.')[0]

    label_files = []
    shapes = []

    for file in os.listdir(labels_path):
        if fnmatch.fnmatch(file, img_name+'*'):
            label_files.append(file)

    for label_file_path in label_files:

        label_file = open(resource_path(os.path.join(labels_path,label_file_path)),'r')
        
        max_confidence_score = 0

        for i,line in enumerate(label_file):
            clss, y, x, width, height , confidence_score = line.split(' ')   
            max_confidence_score = max(float(max_confidence_score),float(confidence_score.strip()))
         
        label_file.close()
        
        label_file = open(os.path.join(labels_path,label_file_path),'r')

        for i,line in enumerate(label_file):
            clss, y, x, width, height , confidence_score = line.split(' ')      
            
            if float(confidence_score.strip()) == max_confidence_score: 

                x = float(x)
                y = float(y)
                width = float(width)
                height = float(height)
                clss = int(clss)

                rel_x_location = round(actual_width*x)
                rel_y_location = round(actual_height*y)
                rel_width = round(actual_width*width)
                rel_height = round(actual_height*height)
                
                
                shape = color_rectangle(clss, np_img, rel_x_location, rel_y_location, rel_width, rel_height)
                json_saver["shapes"].append(shape)

    json_object = json.dumps(json_saver) 

    json_save_path = resource_path(os.path.join(temp_save_dir,img_name+'_ir.json'))
    with open(json_save_path,'w') as f:
        f.write(json_object)
    
    return  json_save_path


