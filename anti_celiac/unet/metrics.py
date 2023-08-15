from PIL import Image
from anti_celiac.unet.utils import draw_boxes
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, cosine_similarity, mean_squared_error
from anti_celiac.unet.utils import gradient_edges

smooth = 1.

def interpretable_iou(y_true, y_pred, domain_mask):
    """
    y is the list of boxes (x1, y1, x2, y2)
    domain_mask is a binary mask of regions to consider in it (PIL object)
    """
    gt_mask = draw_boxes(Image.new('L', domain_mask.size), y_true, "#ffffff", fill="#ffffff")
    pred_mask = draw_boxes(Image.new('L', domain_mask.size), y_pred, "#ffffff", fill="#ffffff")
    gt_mask = (np.array(gt_mask) // 255) * (np.array(domain_mask) // 255)
    pred_mask = (np.array(pred_mask) // 255) * (np.array(domain_mask) // 255)
    intersection = np.sum(gt_mask * pred_mask) + 1.
    union = np.sum(np.clip(gt_mask+pred_mask, 0, 1)) + 1.
    return intersection / union


def dice_coef(y_true, y_pred):
    y_true_c = tf.cast(y_true, tf.float32)
    y_true_f = tf.keras.layers.Flatten()(y_true_c)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.math.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(factor):
    def d_loss(y_true, y_pred):
        return factor * (1.0 - dice_coef(y_true, y_pred))
    return d_loss

def bce_dice_loss(y_true, y_pred):
    return 0.2 * binary_crossentropy(y_true, y_pred) + 0.8 * dice_loss(y_true, y_pred)

def focal_loss(y_true, y_pred):
    # y_pred = y_pred[..., :5]

    alpha = 0.25
    gamma = 2
    # print(y_true[..., -1].shape)
    # print(y_pred[..., -1].shape)
    y_true = tf.reshape(y_true[..., -1], [-1, y_pred.shape[1]*y_pred.shape[2]])
    y_pred = tf.reshape(y_pred[..., -1], [-1, y_pred.shape[1]*y_pred.shape[2]])

    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

def tversky(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    y_true_pos = K.flatten(y_true_f)
    y_pred_pos = K.flatten(y_pred_f)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    ti = tversky(y_true, y_pred)
    gamma = (1.0)
    return K.pow((1-ti), gamma)

def multiclass_dice_coeff(y_true, y_pred):
    """
    both tensors are [b, h, w, classes]
    returns a tensor with dice coeff values for each class
    """
    y_true_c = tf.cast(y_true, tf.float32)
    y_true_shape = tf.shape(y_true_c)
    # [b, h*w, classes]
    y_true_f = tf.reshape(y_true_c, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred_f = tf.reshape(y_pred, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    
    intersection = tf.math.multiply(y_true_f, y_pred_f)
    # [1, classes]
    intersection = 2 * tf.reduce_sum(intersection, axis=[0, 1]) + smooth
    total = tf.reduce_sum(y_true_f, axis=[0, 1]) + tf.reduce_sum(y_pred_f, axis=[0, 1]) + smooth

    return tf.math.divide(intersection, total)


def multiclass_dice_loss(loss_scales):
    total = np.sum(np.array(loss_scales))
    loss_scales = tf.convert_to_tensor(loss_scales, dtype=tf.float32)
    def md_loss(y_true, y_pred):
        return (total - tf.math.reduce_sum(loss_scales * multiclass_dice_coeff(y_true, y_pred))) / total
    return md_loss

def ch_dice_coeff(channel):
    def dice(y_true, y_pred):
        y_true = tf.slice(y_true, [0, 0, 0, channel], [-1, -1, -1, 1])
        y_pred = tf.slice(y_pred, [0, 0, 0, channel], [-1, -1, -1, 1])
        return multiclass_dice_coeff(y_true, y_pred)
    return dice



def total_squared_error(y_true, y_pred):
    return K.sum((y_true - y_pred)**2)

def bifurcated_mse(y_true, y_pred):
    """
    y_true and y_pred of shape (B, grid_w, grid_h, 5)
    sum of mse(pos samples) and mse(neg samples)
    """
    y_pred = y_pred[..., :5]

    labels = y_true[..., -1]
    y_true = y_true[..., 0:4]
    y_pred = y_pred[..., 0:4]
    sq_err = (y_true - y_pred) ** 2
    sq_err = tf.reduce_sum(sq_err, axis=-1) # (B, gw, gh)
    log_sq_err = tf.math.log(1. + sq_err)
    pos_loss = tf.where(labels == 1, log_sq_err, 0)
    neg_loss = tf.where(labels == 0, log_sq_err, 0)

    # loss = 2 * tf.reduce_mean(pos_loss) + 0.5 * tf.reduce_mean(neg_loss)
    loss = tf.reduce_mean(pos_loss)
    return loss

def focal_mse_loss(y_true, y_pred):
    """
    y_true and y_pred of shape (B, grid_w, grid_h, 5)
    """
    y_pred = y_pred[..., :5]
    labels = y_true[..., -1]
    bboxes = y_true[..., 0:4]
    classification = y_pred[..., -1]
    regression = y_pred[..., 0:4]

    # Compute focal loss for classification labels
    cls_loss = focal_loss(y_true, y_pred)

    # Regression loss is mean mse of null boxes + mean mse of active boxes
    reg_loss = bifurcated_mse(y_true, y_pred)

    return cls_loss + reg_loss


def constraint_focal_mse_loss(input_shape, scale, batch_size, factor):
    """
    masks is an Input tensor of shape (B, w, h, num_masks)
    """
    num_boxes = int(input_shape[0]//scale) * int(input_shape[1]//scale)
    def _constraint_focal_mse(y_true, y_pred):
        masks = tf.reshape(y_pred[..., 5:], [batch_size, input_shape[0], input_shape[1], 3])
        y_pred = y_pred[..., :5]

        labels = y_true[..., -1]
        bboxes = y_true[..., 0:4]
        classification = y_pred[..., -1]
        regression = y_pred[..., 0:4]

        # x and y are the i, j values of the input image 2d array
        x_arr = tf.range(start=0, limit=input_shape[0], delta=1, dtype=tf.float32)
        y_arr = tf.range(start=0, limit=input_shape[1], delta=1, dtype=tf.float32)
        x, y = tf.meshgrid(x_arr, y_arr, indexing='ij')
        x, y = tf.stack([x]*num_boxes, axis=-1), tf.stack([y]*num_boxes, axis=-1)
        x, y = tf.stack([x]*batch_size, axis=0), tf.stack([y]*batch_size, axis=0)

        # i and j are the i, j values of the output grid 2d array
        i_arr = tf.range(start=0, limit=input_shape[0]//scale, delta=1, dtype=tf.float32)
        j_arr = tf.range(start=0, limit=input_shape[1]//scale, delta=1, dtype=tf.float32)
        i, j = tf.meshgrid(i_arr, j_arr, indexing='ij')

        # Currently regression tensor is in encoded format (cx - scale*i, cy - scale*j, w, h)
        cx = regression[..., 0]
        cy = regression[..., 1]
        cx = cx + scale * i
        cy = cy + scale * j
        w = regression[..., 2]
        w = tf.where(w == 0, 1., w)
        h = regression[..., 3]
        h = tf.where(h == 0, 1., h)

        # x1, y1 are top left corners of boxes
        x1 = cx - w/2
        x1 = tf.reshape(x1, [batch_size, 1, 1, num_boxes])
        y1 = cy - h/2
        y1 = tf.reshape(y1, [batch_size, 1, 1, num_boxes])
        w = tf.reshape(w, [batch_size, 1, 1, num_boxes])
        h = tf.reshape(h, [batch_size, 1, 1, num_boxes])

        # tensor (B, w_img, h_img, num_boxes) denoting constraint sat
        alpha_x = (x - x1)/w
        alpha_y = (y - y1)/h
        marked = tf.cast(alpha_x >= 0., tf.float32) * tf.cast(alpha_x <= 1., tf.float32) * tf.cast(alpha_y >= 0., tf.float32) * tf.cast(alpha_y <= 1., tf.float32)

        constraint0 = marked * tf.expand_dims(masks[..., 0], axis=-1)
        sat0 = tf.math.reduce_sum(constraint0, [1, 2])
        sat0 = tf.clip_by_value(sat0, 0., 1.)
        sat0 = tf.reshape(sat0, [batch_size, int(input_shape[0]//scale), int(input_shape[1]//scale)])

        constraint1 = marked * tf.expand_dims(masks[..., 1], axis=-1)
        sat1 = tf.math.reduce_sum(constraint1, [1, 2])
        sat1 = tf.clip_by_value(sat1, 0., 1.)
        sat1 = tf.reshape(sat1, [batch_size, int(input_shape[0]//scale), int(input_shape[1]//scale)])

        constraint2 = marked * tf.expand_dims(masks[..., 2], axis=-1)
        sat2 = tf.math.reduce_sum(constraint2, [1, 2])
        sat2 = tf.clip_by_value(sat2, 0., 1.)
        sat2 = tf.reshape(sat2, [batch_size, int(input_shape[0]//scale), int(input_shape[1]//scale)])

        sat = (sat0 * sat1 * sat2)
        sat = tf.where(labels == 1, sat, 1.)
        sat = tf.expand_dims(sat, axis=-1)
        
        # calculate losses
        cls_loss = focal_loss(y_true, y_pred) + factor * focal_loss(y_true * (1. - sat), y_pred * (1. - sat))
        reg_loss = bifurcated_mse(y_true, y_pred) + factor * bifurcated_mse(y_true * (1 - sat), y_pred * (1 - sat))

        loss = cls_loss + reg_loss
        return loss
    
    return _constraint_focal_mse


def cosine_loss(y_true, y_pred):
    labels = y_true[..., -1]
    y_true = y_true[..., 0:4]
    y_pred = y_pred[..., 0:4]

    cos_loss = 1. + cosine_similarity(y_true, y_pred, axis=-1)
    cos_loss = tf.where(labels == 1, cos_loss, 0.)

    loss = tf.reduce_mean(cos_loss)
    return loss


# def centroid_loss(y_true, y_pred):
#     """
#     y_true and y_pred of shape (B, grid_w, grid_h, 5)
#     """
#     labels = y_true[..., -1]
#     bboxes = y_true[..., 0:4]
#     classification = y_pred[..., -1]
#     regression = y_pred[..., 0:4]

#     cls_loss = focal_loss(y_true, y_pred)
#     reg_loss = mean_squared_error()

def uniclass_dice_coeff_0(y_true, y_pred):
    y_true0 = y_true[..., 0:1]
    y_pred0 = y_pred[..., 0:1]
    return multiclass_dice_coeff(y_true0, y_pred0)

def uniclass_dice_coeff_1(y_true, y_pred):
    y_true1 = y_true[..., 1:2]
    y_pred1 = y_pred[..., 1:2]
    return multiclass_dice_coeff(y_true1, y_pred1)

def uniclass_dice_coeff_2(y_true, y_pred):
    y_true2 = y_true[..., 2:3]
    y_pred2 = y_pred[..., 2:3]
    return multiclass_dice_coeff(y_true2, y_pred2)

def uniclass_dice_coeff_3(y_true, y_pred):
    y_true3 = y_true[..., 3:4]
    y_pred3 = y_pred[..., 3:4]
    return multiclass_dice_coeff(y_true3, y_pred3)

def uniclass_dice_coeff_4(y_true, y_pred):
    y_true4 = y_true[..., 4:5]
    y_pred4 = y_pred[..., 4:5]
    return multiclass_dice_coeff(y_true4, y_pred4)


def edge_weighted_focal_tversky_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # tf.print('y_true', y_true)
    # tf.print('y_pred', y_pred)
    
    edge_true = gradient_edges(y_true)
    edge_pred = gradient_edges(y_pred)
    # tf.print('edge_true', edge_true)
    # tf.print('edge_pred', edge_pred)

    ft_loss = focal_tversky_loss(y_true, y_pred)
    edge_loss = focal_tversky_loss(edge_true, edge_pred)
    
    # tf.print('ftl', ft_loss)
    # tf.print('edl', edge_loss)

    return (1. * ft_loss) + (1. * edge_loss)
