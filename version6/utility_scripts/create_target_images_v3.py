import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from image_bboxes import bboxes_of_image, aggregation_func
import random, os


def create_target_image(image_label, dataframe, dim):
    
    """
    Difference with v1:
    v1: creates target image of the same dimensions as the image file and then resizes it.
    This causes issues with the pixel values to be trained, i.e. original target image pixel values of 1,2,3
    change to 1.25,1.5,1.75,2.0, 2.5 and 3.0.
    
    In v2: resize the target image first and then draw the bboxes with appropriate pixel values of 1,2,3
    
    Creates a target image with boxes (pixel value =1) drawn, inside filled with pixel value=2
    and outside 3
    :param image_label: string. name of the image file without the format ending, string
    :param dataframe: pandas dataframe. the dataframe containing the bbox values
    :param dim: int. target image size
    :return: 3D tensor. returns target image as a tensor resized to the desired size
    """
    
    # Original image dimensions.
    original_dim = 1024
    # to be changed to
    
    width = dim
    height = dim
    
    
    
    # Collecting all bbox values in one object
    box_val = bboxes_of_image(image_label, dataframe) # returns an array of arrays of shape: (no. of boxes,)
    box_val = aggregation_func(box_val) # recompiles into an array of proper shape: (no. of boxes x 4)
    
    # Resizing box_val
    if dim == original_dim:
        pass
    else:
        box_val = box_val*dim/original_dim
       
    
    # blank target image
    blank_img = np.full(shape=(height, width), fill_value=0.0, dtype=np.float32)
    
    # drawing bboxes
    trg_img = tf.image.draw_bounding_boxes(
        images = blank_img.reshape(1, height,width,1),
        boxes=tf.reshape(tf.convert_to_tensor(box_val/dim), shape = (1,-1,4)),
        colors = np.array([[1.0, 1.0, 0.0]]))
    
    
        
    
    trg_img = trg_img.numpy().reshape(height,width)
    # assigning value 2 to pixels inside bboxes
    for i in box_val:
        yy = np.arange(i[0], i[2], dtype=np.int16)
        xx = np.arange(i[1], i[3], dtype=np.int16)
    
        yy, xx = np.meshgrid(yy, xx)
    
        # trg_img[yy, xx] = np.where(trg_img[yy, xx] == 1.0, 2.0, 1.0)
        trg_img[yy, xx] = 1.0
    
    trg_img =  np.expand_dims(trg_img, axis=-1)
    
    
    return trg_img # 3D array
    


if __name__=='__main__':
    
    train_dir = 'train/'
    
    img_file_list = os.listdir(train_dir)
    
    # choosing random image
    random_imag = random.choice(img_file_list)
    
    test_img = image.load_img(os.path.join(train_dir, random_imag))
    test_img_arr = image.img_to_array(test_img)
    img_arr_shape = test_img_arr.shape
    
    # bbox data
    train_bbox_data = pd.read_csv('train.csv', header=0)

    # returns the bboxes of an image for cross checking with the target bbox image
    test_bbox_values = bboxes_of_image(random_imag[:-4], train_bbox_data)
    test_bbox_values = aggregation_func(test_bbox_values)
    
    # drawing bboxes on the test image for comparing with the target image
    bbox_img = tf.image.draw_bounding_boxes(
        test_img_arr.reshape(1, img_arr_shape[0], img_arr_shape[1], img_arr_shape[2]),
        boxes=tf.reshape(tf.convert_to_tensor(test_bbox_values / img_arr_shape[0]),
                         shape=(1, -1, 4)), colors=np.array([[1.0, 1.0, 0.0]]))

    bbox_img = image.array_to_img(tf.reshape(bbox_img, shape=(img_arr_shape[0],
                                                              img_arr_shape[1], img_arr_shape[2])))
    bbox_img.show()
    
    # target size
    trg_size = 256
    trgt_img = create_target_image(random_imag[:-4], train_bbox_data, trg_size)
    print(np.unique(trgt_img, return_counts=True))
    pil_img = image.array_to_img(trgt_img)
    pil_img.show()
    
    
    