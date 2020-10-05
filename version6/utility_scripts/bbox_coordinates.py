import numpy as np
import  pandas as pd
import tensorflow as tf


def return_edge_val_along_rows(arr = np.array([])):
    """
    scanning values of each row of the predicted y array
    arr: shape: y shape: rows x columns

    returns: list containing the indices of the edges of all the bboxes scanned row by row of the predicted image
    """
    
    dim = arr.shape[1]
    #     print(dim)
    
    scanlist = []
    
    for i, pixval in enumerate(arr):  # for each row in the predicted array
        #         print(i)
        scan_ind = np.where(pixval == 1)[0]  # returns the indices containing the bbox
        
        if len(scan_ind) == 0:
            continue  # if the row has no bbox values
        else:
            templist = []
            templist.append(scan_ind[0])  # append the first instance or the left edge value
            
            for j in range(len(scan_ind) - 1):
                
                if scan_ind[j + 1] - scan_ind[j] == 1:
                    # looking for gaps between the bboxes where the difference between the indices
                    # is more than 1
                    
                    if (scan_ind[j + 1] == dim - 1) | (scan_ind[j + 1] == scan_ind[-1]):
                        # when the edge of the image or the right edge
                        # of the last bbox along the row is reached
                        templist.append(scan_ind[j + 1])
                    else:
                        continue
                else:
                    # if the difference between the indices is more than 1, then
                    templist.append(scan_ind[j])  # right edge of one bbox
                    templist.append(scan_ind[j + 1])  # left edge of the next bbox
            
            bbox_columnscan = np.full(shape=(len(templist), 2), fill_value=i)  # keeping track of the row number
            bbox_columnscan[:, 1] = templist  # with row number added, saving the edge values of the bboxes
            
            #         print(bbox_columnscan)
            
            scanlist.append(bbox_columnscan.tolist())
            # appending the edge indices of all the bboxes found the predicted mask
    
    return scanlist

def edge_coordinates(scannedlist=[]):
    """
    
    :param scannedlist: list returned from the return_edge_val_along_rows function
    :return: DataFrame: bbox coordinates: [ymin, xmin, ymax, xmax]
    """
    
    flattened_arr = [elem for list_ in scannedlist for elem in list_]
    # flattening the list
    flattened_arr = np.asarray(flattened_arr).reshape(-1, 4)
    flattened_arr = flattened_arr[:,[0,1,3]]
    flattened_list = flattened_arr.tolist()
    
    flattened_list.sort(key = lambda x:x[1])
    flattened_list.sort(key = lambda x:x[2])
    # sorting based on 1. height, 2. width
    
    df = pd.DataFrame(flattened_arr, columns = ['row', 'col1', 'col2'])
    df_min = df.drop_duplicates(subset=['col1', 'col2'], keep='first').reset_index(drop=True)
    # keeping the first row value
    df_max = df.drop_duplicates(subset=['col1', 'col2'], keep='last').reset_index(drop=True)
    # keeping the last row value
    df_merged = pd.merge(df_min, df_max, on=['col1', 'col2'])
    # merging
    
    collist = df_merged.columns.tolist()
    # rearranging columns to have the order: ymin, xmin, ymax, xmax
    correct_order_collist = collist[:2] + collist[-1:] + collist[-2:-1]
    df_merged = df_merged[correct_order_collist]
    
    return  df_merged.values
    
    
def draw_img_with_bbox(imag_arr = np.array([]), boxarr = np.array([])):
    """
    :param imag_arr:
    :param boxarr: box edge coordinates returned from the function edge_coordinates
   
    :return: image with bbox drawn
    """
    dim = imag_arr.shape[1]
    imag_with_box = tf.image.draw_bounding_boxes(np.expand_dims(imag_arr, axis=0),
                                                 boxes=np.expand_dims(boxarr, axis=0)/(dim-1),
                                                 colors=np.array([[1.0, 1.0, 0.0]]))
    
    imag_with_box = np.squeeze(imag_with_box.numpy(), 0)
    return imag_with_box



# if __name__=='__main__':

    
    



