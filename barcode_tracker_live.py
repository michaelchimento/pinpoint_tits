#!/bin/bash/
import cv2
import numpy as np
import datetime as dt
import time as time
from scipy.spatial import distance as dist
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import re


def add_border(tag, tag_shape, white_width = 1, black_width = 2):

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


def crop(src, pt1, pt2):

    """ Returns a cropped version of src """

    cropped = src[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    return cropped

def distance(vector):

    """ Return distance of vector """

    return np.sqrt(np.sum(np.square(vector)))

def order_points(pts):
    # sort the points based on their x-coordinates
    sorted_ = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left = sorted_[:2, :]
    right = sorted_[2:, :]

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

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = np.subtract(A, A.mean(1)[:,None])
    B_mB = np.subtract(B, B.mean(1)[:,None])

    # Sum of squares across rows
    ssA = np.square(A_mA).sum(1)
    ssB = np.square(B_mB).sum(1)

    # Finally get corr coeff
    return np.divide(np.dot(A_mA,B_mB.T), np.sqrt(np.dot(ssA[:,None],ssB[None])))

def unit_vector(vector):

    """ Returns the unit vector of the vector.  """

    return np.divide(vector, np.linalg.norm(vector))

def angle(vector, degrees = True):

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

    #v1_u = unit_vector(v1)
    #v2_u = unit_vector(v2)
    angle = np.arctan2(vector[1], vector[0]) % (2*np.pi)
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    if degrees == True:
        angle = np.degrees(angle)
    return angle

def get_grayscale(color_image, channel = None):

    """ Returns single-channel grayscale image from 3-channel BGR color image.

        Parameters
        ----------
        color_image : (MxNx3) numpy array
            3-channel BGR-format color image as a numpy array
        channel : {'blue', 'green', 'red', 'none', None}, default = None
            The color channel to use for producing the grayscale image.
            None and 'none' default to cv2.cvtColor() using cv2.COLOR_BGR2GRAY.
            Channels 'blue', 'green', and 'red' use the respective color channel as the grayscale image.
            Channel 'green' typically provides the lowest noise, but this will depend on the lighting in the image.

        Returns
        -------
        gray_image : (MxNx1) numpy array
            Single-channel grayscale image as a numpy array.

    """
    assert channel in ['blue', 'green', 'red', 'none', None], "channel must be 'blue', 'green', 'red', 'none', or None"
    assert type(color_image) == np.ndarray, "image must be a numpy array"

    image_shape = color_image.shape
    assert len(image_shape) == 3, "image must be color"
    assert color_image.shape[2] == 3, "image must have 3 color channels"
    assert color_image.dtype == np.uint8, "image array must be dtype np.uint8"

    if channel == 'blue':
        gray_image, _, _ = cv2.split(color_image)
    if channel == 'green':
            _, gray_image, _ = cv2.split(color_image)
    if channel == 'red':
        _, _, gray_image = cv2.split(color_image)
    if channel == None or channel == 'none':
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    return gray_image

def get_threshold(gray_image, block_size = 1001, offset = 2):

    """ Returns binarized thresholded image from single-channel grayscale image.

        Parameters
        ----------
        gray_image : (MxNx1) numpy array
            Single-channel grayscale image as a numpy array
        block_size : int, default = 1001
            Odd value integer. Size of the local neighborhood for adaptive thresholding.
        offset : default = 2
            Constant subtracted from the mean. Normally, it is positive but may be zero or negative as well.
            The threshold value T(x,y) is a mean of the block_size x block_size neighborhood of (x, y) minus offset.

        Returns
        -------
        threshold_image : (MxNx1) numpy array
            Binarized (0, 255) image as a numpy array.

    """

    assert block_size % 2 == 1, "block_size must be an odd value"
    assert type(gray_image) == np.ndarray, "image must be a numpy array"

    assert len(gray_image.shape) == 2, "image must be grayscale"
    assert gray_image.dtype == np.uint8, "image array must be dtype np.uint8"

    threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, offset)

    return threshold_image

def get_contours(threshold_image):

    """ Returns a list of contours from a binarized thresholded image.

        Parameters
        ----------
        threshold_image : (MxNx1) numpy array
            Binarized threshold image as a numpy array

        Returns
        -------
        contours : list
            List of contours extracted from threshold_image.

    """
    #assert len(set([0, 255]) - set(np.unique(threshold_image))) == 0, "image must be binarized to (0, 255)"

    contours, _ = cv2.findContours(threshold_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def contour_loop(contours, image, dst, gray, maxSide, barcode_size, barcodes, flat_len, IDs, font,timeofframe, frame, pt1, population):
    # define frame edges for checking for tags
    edge_thresh = 1
    image_shape = image.shape
    maxx_thresh = image_shape[1] - edge_thresh
    maxy_thresh = image_shape[0] - edge_thresh
    to_write_list = []
    upper_size_limit = -11000
    lower_size_limit = -4500

    for cnt in contours:

        cnt_shape = cnt.shape

        if cnt_shape[0] >= 4: # only look at contours that could be tags

            area = cv2.contourArea(cnt, True)


            if lower_size_limit > area > upper_size_limit: # check area
                cv2.drawContours(image, [cnt], -1, (0,0,255), 2)

        # check for contours too close to the edge to read
                cnt_reshape = cnt.reshape((cnt_shape[0], cnt_shape[2]))
                cnt_x = cnt_reshape[:,0]
                cnt_y = cnt_reshape[:,1]
                flat = cnt.flatten()
                edge_zero = np.sum(flat <= edge_thresh)
                edge_maxx = np.sum(cnt_x >= maxx_thresh)
                edge_maxy = np.sum(cnt_y >= maxy_thresh)

                if edge_zero + edge_maxx + edge_maxy == 0:

                    # fit a polygon
                    peri_cnt = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.07 * peri_cnt, True)
                    poly_area = cv2.contourArea(approx, True)

                    # check if it's approximately a parallelogram
                    if 4 <= len(approx) <= 5 and lower_size_limit > poly_area > upper_size_limit and cv2.isContourConvex(approx) and 0 < peri_cnt < 600:
                        periarea = peri_cnt/area
                        cv2.drawContours(image, [cnt], -1, (0,0,255), 2)
                        # check that the geometry isn't too complex
                        if -0.5 < periarea <= 0:
                            #cv2.drawContours(image, [cnt], -1, (0,255,255), 1)
                            cnt_shape = approx.shape
                            pts = approx.reshape((cnt_shape[0], cnt_shape[-1]))

                            # get the corners of the parallelogram
                            pts = order_points(pts)
                            (tl, tr, br, bl) = pts

                            # compute the perspective transform matrix and then apply it
                            M = cv2.getPerspectiveTransform(pts, dst)
                            warped = cv2.warpPerspective(gray, M, (maxSide, maxSide), borderValue = 255 )
                            resize_warp = cv2.resize(warped, barcode_size, interpolation = cv2.INTER_AREA)

                            # calculate best match with master_list
                            correlation = corr2_coeff(barcodes, resize_warp.reshape((1,flat_len)))
                            best_value = np.max(correlation)

                            if best_value > 0.8: #check for prob of match
                                best_index = np.argmax(correlation)
                                ID = IDs[best_index]
                                centroid = np.array(pts.mean(0))
                                y_offset = 0
                                x_offset = 0
                                bottom_centroid = tuple((centroid + np.array([x_offset,-1*y_offset])).astype(int))
                                top_centroid = tuple((centroid + np.array([x_offset,y_offset])).astype(int))
                                mid_centroid = tuple((centroid + np.array([x_offset,0])).astype(int))
                                rotate_test = best_index % 4

                                if rotate_test == 3:
                                    edge = np.array(np.mean([tl, tr], axis = 0))
                                if rotate_test == 0:
                                    edge = np.array(np.mean([tl, bl], axis = 0))
                                if rotate_test == 1:
                                    edge = np.array(np.mean([br, bl], axis = 0))
                                if rotate_test == 2:
                                    edge = np.array(np.mean([br, tr], axis = 0))

                                cv2.drawContours(image, [approx], -1, (255,0,0), 1)

                                edge[1] = -edge[1]
                                centroid[1] = -centroid[1]
                                vector = np.subtract(edge, centroid)
                                vector_angle = angle(vector)
                                angle_str = '%.0f' % vector_angle
                                bestval_str = '%.2f' % best_value
                                font_scale = 1.5
                                outline_font = 5
                                inline_font = 2

                                cv2.putText(image,str(ID),mid_centroid, font, font_scale,(0,0,0),outline_font,cv2.LINE_AA)
                                cv2.putText(image,str(ID),mid_centroid, font, font_scale,(255,255,255),inline_font,cv2.LINE_AA)

                                #write to data file
                                to_write_line = "{},{},{},{},{},{},{},{}\n".format(population, timeofframe, frame, ID, best_value, (centroid[0]+pt1[0]), (centroid[1]+pt1[1]), vector_angle )
                                to_write_list.append(to_write_line)

    num_detections = len(to_write_list)

    detected_tags = "".join(to_write_list)

    return detected_tags, num_detections


def write_csv(to_write_list, data_filepath):
    with open(data_filepath, "a") as savefile: # open data file in write mode
        # write column names to file
        savefile.write(to_write_list)

def decode(video_filepath,data_filepath,tags,target_pop):
    master_list = tags.master_list
    IDs = tags.id_list
    zipped_list = list(zip(IDs, master_list))
    if target_pop=="P1":
        approved_list = [32,7,14,23,56,52]
    elif target_pop=="P2":
        approved_list = [6,5,22,44,38,31]
    elif target_pop=="P3":
        approved_list = [107,120,95,103,2,3,35,10,17,65,93,85,122,125]
    elif target_pop=="P4":
        approved_list = [82,4,33,13,18,28,45,84,94,109,112,108,121,126]
    elif target_pop=="P5":
        approved_list = [34,15,9,16,42,37]
    elif target_pop=="P6":
        approved_list = [123,127,75,50,46,69,61,66,87,86,104,106,116,118]
    elif target_pop=="P7":
        approved_list = [124,73,59,24,41,39,53,80,83,96,98,102,114,117,128]
    elif target_pop=="P8":
        approved_list = [36,30,27,40,63,37]
    elif target_pop=="P9":
            approved_list = [129,130,12,72,54,47,67,19,81,89,88,110,101,113,119]
    elif target_pop=="P10":
        approved_list = [76,49,48,64,58,55,58,90,91,131,100,105,111,115,132]
    elif target_pop=="P11":
        approved_list = [8,43,51,11,20,60]
    elif target_pop=="P12":
        approved_list = [25,74,68,62,29,71]

    IDs = [id[0] for id in zipped_list if id[0] in approved_list]
    barcode_size = (7,7)
    barcodes = []
    for barcode in zipped_list:
        if barcode[0] in approved_list:
            tag_shape, barcode = add_border(barcode[1], (5,5), white_width = 1, black_width = 0)
            barcode = barcode.reshape(tag_shape)
            barcode = cv2.resize(barcode, barcode_size, interpolation = cv2.INTER_AREA)
            barcode = barcode.flatten()
            barcodes.append(barcode)
    barcodes = np.array(barcodes)
    assert len(barcodes)==len(IDs), "id list does not equal barcode list"

    # set various parameters for tracker subroutines
    flat_len = barcode_size[0]*barcode_size[1]
    maxSide = 100
    length = maxSide - 1

    dst = np.array([
                [0, 0],
                [length, 0],
                [length, length],
                [0, length]], dtype = "float32")

    cap = cv2.VideoCapture(video_filepath)

    assert cap.isOpened(), "video didn't open properly"

    #set cropped area, default is full frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #frame width as integer
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#frame height as integer
    pt1 = (0,0) #top-left corner
    pt2 = (frame_width,frame_height) #bottom-right corner
    #pt2 = (1920,1080) #bottom-right corner

    font = cv2.FONT_HERSHEY_SIMPLEX # set font
    frame = 0
    process_every_nth_frame = 3

    cv2.namedWindow("preview")

    #regex search datetime in filename and set that as the start time
    population = re.search("P\d?\d", video_filepath).group(0)[1:]
    file_timestamp= re.search("\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d", video_filepath)
    print(file_timestamp.group(0))
    start_time = dt.datetime.strptime(file_timestamp.group(0),"%Y-%m-%d_%H-%M-%S")
    print(start_time)

    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            if frame%process_every_nth_frame == 0:
                #if we process every 3th frame of a 30 fps video, we have a time delta between readings of 100 ms.
                timeofframe = start_time + dt.timedelta(milliseconds=100)*(frame/process_every_nth_frame)
                gray = get_grayscale(image, channel = 'green')
                gray = cv2.GaussianBlur(gray, (1,1), 1)
                thresh = get_threshold(gray, block_size = 21, offset = 2)
                contours = get_contours(thresh)

                #this begins to loop through the detected contours
                detected_tags, num_detections = contour_loop(contours, image, dst, gray,
                                             maxSide, barcode_size, barcodes,
                                             flat_len, IDs, font, timeofframe, frame, pt1,population)

                #print("num_detections:{}".format(num_detections))
                if num_detections > 0:
                    print(num_detections)
                    write_csv(detected_tags, data_filepath)

            cv2.imshow("preview",image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            frame += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
