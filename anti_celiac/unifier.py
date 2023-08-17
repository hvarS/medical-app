import os 

def uniform_path(img_path):
    if os.name=='nt':
        return '/'.join(img_path.split('\\'))
    else:
        return img_path