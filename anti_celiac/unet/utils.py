from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import json
import tensorflow as tf
import cv2

def load_image(img_path):
    """
    Load an image at the index.
    Returns PIL image
    """
    img = Image.open(img_path).convert('RGB')
    # img = np.array(img)
    # print("Loaded image: ", image_path)
    return img


def load_annotations(anns_path):
    """
    Load annotations for an image_index.
    """
    labels = json.load(open(anns_path))
    labels = labels["shapes"]
    return labels.copy()


def get_ground_truth(img_path, anns_path, target_classes):
    """
    Input: The instance id path
    Output: The image and the ground truth binary mask images (PIL)
    """
    img = load_image(img_path)
    anns = load_annotations(anns_path)
    image_size = img.size

    masks = {label: Image.new("L", image_size) for label in target_classes}
    draw_masks = {label: ImageDraw.Draw(masks[label]) for label in target_classes}
    # Get valid present annotations
    anns = [x for x in anns if type(x) == dict]

    # Gathering polygon points - A dict of points for each target class
    poly_pts = {label: [] for label in target_classes}
    for ann in anns:
        if ann['label'] in target_classes:
            poly_pts[ann['label']].append([np.rint(point) for point in ann['points']])

    # Drawing the masks
    for label in target_classes:
        polygons = poly_pts[label]
        for poly in polygons:
            coords = [(pt[0], pt[1]) for pt in poly]
            draw_masks[label].polygon(xy=coords, fill=255)
    
    return img, masks


def infer_image_with_anns(img_path, anns_path, image_size, target_classes, model, conf_threshold):
    """
    Inputs: the path to image, its annotations, (W,H), list of classes, preprocessing func, keras model object, confidence threshold
    Returns: The original image, the result mask image, combined 1 & 2, combined ground truth
    """
    raw_image, gt_masks = get_ground_truth(img_path, anns_path, target_classes)
    w, h = raw_image.size
    img = raw_image.resize(image_size)
    img = np.array(img)
    # Convert (H, W, C) to (W. H, C)
    img = np.transpose(img, (1, 0, 2))
    img = img.astype(np.float32)
    img = img/255.0
    # Make a batch of 1 image
    test_image = tf.convert_to_tensor(np.array([img]))

    result_mask = model.predict(test_image, verbose=1)
    if type(result_mask) == list:
        result_mask = result_mask[0]
    result_mask = (result_mask > conf_threshold) * 255
    result_mask = np.transpose(result_mask, (0, 2, 1, 3))
    if(len(target_classes) != 3):
        dummy_shape = result_mask.shape
        dummy_shape[-1] = 3 - dummy_shape[1]
        dummy = np.zeros(dummy_shape)
        result_mask = np.concatenate([result_mask, dummy], axis=-1)
    result_mask = Image.fromarray(np.uint8(result_mask[0]))

    result_image = np.array(raw_image.resize(image_size))
    result_image += np.uint8((np.array(result_mask) / 255) * 200)
    result_image = np.clip(result_image, 0, 255)
    result_image = Image.fromarray(result_image)

    gt_mask = []
    for label in target_classes:
        gt_mask.append(np.array(gt_masks[label].resize(image_size)))
    gt_mask = np.stack(gt_mask, axis=-1)
    gt_mask = Image.fromarray(gt_mask)

    gt_image = np.array(raw_image.resize(image_size))
    gt_image += np.uint8((np.array(gt_mask) / 255) * 255)
    gt_image = Image.fromarray(gt_image)

    return raw_image.resize(image_size), result_mask, result_image, gt_mask, gt_image


def infer_image(img_path, image_size, target_classes, model, conf_threshold):
    """
    Inputs: the path to image, (W,H), preprocessing func, keras model object, confidence threshold
    Returns: The original image, the result mask image, combined 1 & 2, combined ground truth
    """
    raw_image = load_image(img_path)
    w, h = raw_image.size
    img = raw_image.resize(image_size)
    img = np.array(img)
    # Convert (H, W, C) to (W. H, C)
    img = np.transpose(img, (1, 0, 2))
    img = img.astype(np.float32)
    img = img/255.0
    # Make a batch of 1 image
    test_image = tf.convert_to_tensor(np.array([img]))

    result_mask = model.predict(test_image, verbose=1)
    if type(result_mask) == list:
        result_mask = result_mask[0]
    result_mask = (result_mask > conf_threshold) * 255
    result_mask = np.transpose(result_mask, (0, 2, 1, 3))
    if(len(target_classes) != 3):
        dummy_shape = result_mask.shape
        dummy_shape[-1] = 3 - dummy_shape[1]
        dummy = np.zeros(dummy_shape)
        result_mask = np.concatenate([result_mask, dummy], axis=-1)
    result_mask = Image.fromarray(np.uint8(result_mask[0]))

    result_image = np.array(raw_image.resize(image_size))
    result_image += np.uint8((np.array(result_mask) / 255) * 200)
    result_image = np.clip(result_image, 0, 255)
    result_image = Image.fromarray(result_image)

    return raw_image.resize(image_size), result_mask, result_image


def draw_boxes(image, bboxes, color, fill=None):
    image_d = ImageDraw.Draw(image)
    for box in bboxes:
        x1, y1, x2, y2 = box
        image_d.rectangle([x1, y1, x2, y2], fill=fill, outline=color, width=2)
    return image


def bbox_transform(anchors, gt_boxes):
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + wa / 2.
    cya = anchors[:, 1] + ha / 2.

    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    cx = gt_boxes[:, 0] + w / 2.
    cy = gt_boxes[:, 1] + h / 2.
    # Avoid NaN in division and log below.
    ha += 1e-7
    wa += 1e-7
    h += 1e-7
    w += 1e-7
    tx = (cx - cxa) / wa
    ty = (cy - cya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)
    targets = np.stack([ty, tx, th, tw], axis=1)
    return targets


def compute_overlap(boxes1, boxes2):
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    intersect = intersect_heights * intersect_widths

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect

    return (intersect + 1.) / (union + 1.)


def output2image(result_mask):
    """
    Args: result_mask is image with float values in 0 and 1 (W, H, channels)
    Returns a pil image that can visualize the result_mask
    """
    result_mask_img = result_mask * 255.
    result_mask_img = np.transpose(result_mask_img, (1, 0, 2))
    if result_mask_img.shape[-1] == 4:
        result_mask_img[..., :3] += np.expand_dims(result_mask_img[..., -1], axis=-1) // 1.5
    if result_mask_img.shape[-1] == 5:
        result_mask_img[..., :3] += np.expand_dims(result_mask_img[..., -2], axis=-1) // 1.5  # Brunners gland
        result_mask_img[..., :2] += np.expand_dims(result_mask_img[..., -1], axis=-1)     # Circular crypts
    result_mask_img = np.clip(result_mask_img, 0., 255.)
    result_mask_img = Image.fromarray(np.uint8(result_mask_img[..., :3]))
    return result_mask_img


def histogram_equalization(img):
    R, G, B = cv2.split(img)
    B = cv2.equalizeHist(B.astype(np.uint8))
    G = cv2.equalizeHist(G.astype(np.uint8))
    R = cv2.equalizeHist(R.astype(np.uint8))
    img = cv2.merge((R, G, B))
    return img


def points(image):
    """
    Takes in image: An image array that contains segmentation mask in a target region
    Returns extern box around the total masked blobs in format xmin, ymin, xmax, ymax
    """
    img = np.squeeze(image)
    activations = img > 0
    activations = activations.astype(int)
    nz = np.nonzero(activations)  # A tuple of arrays indicating non zero element indices per axis
    xmin = np.min(nz[1])
    xmax = np.max(nz[1])
    ymin = np.min(nz[0])
    ymax = np.max(nz[0])
    return xmin, ymin, xmax, ymax


def sobel_edges(img):
    """
    img is a tensor between 0. to 1. of shape (B, W, H, C)
    """
    grad_components = tf.image.sobel_edges(img)
    grad_mag_components = tf.math.pow(grad_components, 2)
    grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1)  # sum all magnitude components
    grad_mag_img = tf.sqrt(grad_mag_square)  # this is the image tensor you want
    grad_mag_img = grad_mag_img / (tf.reduce_max(grad_mag_img) + 1e-6)
    return grad_mag_img


def gradient_edges(img):
    """
    img is a tensor between 0. to 1. of shape (B, W, H, C)
    """
    dx, dy = tf.image.image_gradients(img)
    grads = tf.sqrt(dx**2 + dy**2) / tf.sqrt(2.)
    return grads


# def perturb_input(inp, count=5):
#     """
#     inp: input of shape (B, W, H, 3) images. Makes #count random augmentations for same output
#     """
#     aug_inps = []
#     for i in range(count):
#         batch = []
#         for img in inp:
#             img = Image
