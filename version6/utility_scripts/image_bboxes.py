import pandas as pd
from numpy import ones, array, float32
import random
import json
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def reformat_bbox(arr = ones((4,), dtype= float32)):
    """
    :param arr: [xmin, ymin, width, height]
    :return:
    [ymin, xmin, ymax, xmax] fpr tf.image.draw_bounding_box method
    """
    
    
    return array([arr[1], arr[0], arr[1]+arr[3], arr[0]+arr[2]],
                    dtype = float32)


def bboxes_of_image(image_label, dataframe):
    
    """
    
    :param image_label: name of the image file
    :param dataframe: df containing the bbox values of the image file
    :return: array of arrays. bbox values in the order [ymin, xmin, ymax, xmax] on the scale of the
    input image, 1024.
    """

    # choosing rows filtered by the image_label
    image_df = dataframe[dataframe.image_id == image_label]
    
    bbox_index = image_df.columns.tolist().index('bbox')
    # converting bbox of type string to float using json
    image_df.iloc[:, bbox_index] = image_df['bbox'].map(lambda  x:json.loads(x))
    
    # converting the bbox format of xmin, ymin, width, height to
    # the format compatible for draw_bounding_boxes method
    bbox_values = image_df[['bbox']].applymap(lambda  x: reformat_bbox(x)).values.ravel()
    
    return bbox_values # an array of arrays. shape is not correctly defined
    
def aggregation_func(arr):
    """
    
    :param arr: bbox_values from
    :return:
    """
    templist = []
    for i in arr:
        templist.append(i)
        
    return array(templist) # array of the right shape: (n,4)

if __name__=='__main__':
    
    train_dir = 'train/'
    
    img_file_list = os.listdir(train_dir)
    
    random_imag = random.choice(img_file_list)
    
    test_img = image.load_img(os.path.join(train_dir,random_imag))
    test_img_arr = image.img_to_array(test_img)
    img_arr_shape = test_img_arr.shape
    
    # bbox data
    train_bbox_data = pd.read_csv('train.csv', header=0)
    
    # returns the bboxes of an image
    test_bbox_values = bboxes_of_image(random_imag[:-4], train_bbox_data)
    test_bbox_values = aggregation_func(test_bbox_values)
    
    bbox_img = tf.image.draw_bounding_boxes(test_img_arr.reshape(1,img_arr_shape[0], img_arr_shape[1],
                                                                 img_arr_shape[2]),
                                   boxes=tf.reshape(tf.convert_to_tensor(test_bbox_values/img_arr_shape[0]),
                                                 shape = (1,-1,4)), colors = array([[1.0,1.0,0.0]]))
    
    pil_img = image.array_to_img(tf.reshape(bbox_img, shape=(img_arr_shape[0],
                                                             img_arr_shape[1], img_arr_shape[2])))
    pil_img.show()
    
    