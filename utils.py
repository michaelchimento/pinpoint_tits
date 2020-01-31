""" Utility functions for generating and tracking barcode tags"""

import numpy as np
import cv2

from scipy.spatial import distance as dist

def rotate_tag90(tag, tag_shape, n=1):
    
    """Rotate barcode tag 90 degrees.
        
        Parameters
        ----------
        tag : 1-D array_like
            Flattened barcode tag.
        tag_shape : tuple of int
            Shape of the barcode tag.
        n : int
            Number of times to rotate 90 degrees.
        
        
        Returns
        -------
        tag_rot : 1-D array
            Returns rotated tag flattened to 1-D array.
        
        """
    
    tag = np.asarray(tag)
    vector_shape = tag.shape
    tag = tag.reshape(tag_shape)
    tag_rot = np.rot90(tag,n)
    tag_rot = tag_rot.reshape(vector_shape)
    return tag_rot

def add_border(tag, tag_shape, white_width = 1, black_width = 1):
    
    """Add black and white border to barcode tag.
        
        Parameters
        ----------
        tag : 1-D array_like
            Flattened barcode tag.
        tag_shape : tuple of int
            Shape of the barcode tag without a border.
        white_width : int
            Width of white border.
        black_width : int
            Width of black border.
            
        Returns
        -------
        bordered_tag : 1-D array
            Returns tag with border added flattened to 1-D array.
        """
    
    tag = np.asarray(tag)
    tag = tag.reshape(tag_shape)

    black_border = np.zeros((tag_shape[0]+(2*white_width)+(2*black_width),tag_shape[1]+(2*white_width)+(2*black_width)))
    white_border = np.ones((tag_shape[0]+(2*white_width),tag_shape[1]+(2*white_width)))
    
    white_border[white_width:tag_shape[0]+white_width,white_width:tag_shape[1]+white_width] = tag
    black_border[black_width:tag_shape[0]+(2*white_width)+black_width, black_width:tag_shape[1]+(2*white_width)+black_width] = white_border

    tag = black_border
    bordered_tag = tag.reshape((1,tag.shape[0]*tag.shape[1]))
    tag_shape = black_border.shape
    return  tag_shape, bordered_tag

def check_diffs(array1, array2, ndiffs, test_num):
    
    """ Check for differences between two arrays. Each element of array1 is checked against all elements of array2.
        
        Parameters
        ----------
        array1 : 2-D array_like
            Array of flattened barcodes to test against array2.
        array2 : 2-D array_like
            Array of flattened barcodes to test array1 against.
        ndiffs : int
            Minimum number of differences between all barcodes in array1 and array2.
        test_num : int
            Number of elements in array2 that each element in array1 must be at least ndiffs different from.
            
        Returns
        -------
        test : int
            Number of elements in array1 that are at least ndiffs different from all elements in array2.
        
        """
    
    array1 = np.asarray(array1).astype(np.uint8)
    array2 = np.asarray(array2).astype(np.uint8)
    test_list = np.array([], dtype = bool)
    
    for tag in array1:
        tag = np.asarray([tag]).astype(np.uint8)
        repeat_tags = np.repeat(tag, array2.shape[0], axis = 0)
        tag_diffs = np.subtract(array2,repeat_tags)
        tag_diffs[tag_diffs>1] = 1
        tag_diffs = np.sum(tag_diffs, axis = 1)
        diff_bool = tag_diffs >= ndiffs # test for minimum number of differences
        test = np.sum(diff_bool)
            
        if test == test_num: # if element in array1 is different enough from all elements in array2...
            test_list = np.append(test_list, True)
    
    test = np.sum(test_list)
    
    return test

def crop(src, pt1, pt2):
    
    """ Returns a cropped version of src """
    
    cropped = src[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    
    return cropped

def distance(vector):
 
    """ Return distance of vector """

    return np.sqrt(np.sum(np.square(vector)))

def order_points(pts):

    """Sorts a 4x2 array of coordinates clockwise from the top-left corner.

        Parameters
        ----------
        pts : 4x2 array_like
            2-dimensional array of x,y coordinates.
            
        Returns
        -------
        ordered_pts : 4x2 array_like
            Sorted array of x,y coordinates.
        
        """

    # sort the points based on their x-coordinates
    xsorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    left = xsorted[:2, :]
    right = xsorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left = left[np.argsort(left[:, 1]), :]
    (tl, bl) = left
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], right, "euclidean")[0]
    (br, tr) = right[np.argsort(D)[::-1], :]
    
    # return the coordinates in clockwise order 
    # top-left, top-right, bottom-right, bottom-left 
    ordered_pts = np.array([tl, tr, br, bl], dtype="float32")

    return ordered_pts

def rowwise_corr(A,B):

    """Returns the rowwise Pearson product-moment correlation coefficient of a 2-D array and a 1-D vector

        Parameters
        ----------
        A : 2-D array_like
            2-dimensional array.
        B : 1-D array_like
            1-dimensional vector.
            
        Returns
        -------
        corr_coeff : 1-D array_like
            Vector of rowwise correlation coeffecients between A and B.
        
        """

    # Get rowwise error values by subtracting the rowwise mean
    A_error = np.subtract(A, A.mean(axis = 1)[:,None])
    B_error = np.subtract(B, B.mean(axis = 1)[:,None])

    # Get rowwise sum of squaerd errors
    ssA = np.square(A_centered).sum(axis = 1)
    ssB = np.square(B_centered).sum(axis = 1)

    # Get rowwise correlation coefficient 
    corr_coeff = np.divide(np.dot(A_centered,B_centered.T), np.sqrt(np.dot(ssA[:,None],ssB[None])))

    return corr_coeff

def unit_vector(vector):
    
    """ Returns the unit vector of the vector.  """
    
    return np.divide(vector, np.linalg.norm(vector))

def angle(v1, v2, degrees = True):
    
    """Returns the angle between vectors 'v1' and 'v2'.
        
        Parameters
        ----------
        v1 : 1-D array_like
            N-dimensional vector.
        v2 : 1-D array_like
            N-dimensional vector.
        degrees : bool, default = True
            Return angle in degrees.
            
        Returns
        -------
        angle : float
            Angle between v1 and v2.
        
        """
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    if degrees == True:
        angle = np.degrees(angle)
    return angle
