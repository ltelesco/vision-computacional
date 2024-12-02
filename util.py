from os import path
from glob import glob
from configparser import ConfigParser
from visdom import Visdom

import os
from skimage import filters, measure
import re
import cv2
from scipy.ndimage.morphology import binary_fill_holes

from skimage.transform import rescale

import matplotlib.image as mpimg

import numpy as np
from imageio import imread, imwrite
import cv2
from os.path import exists
from PIL import Image

def save_critical_image(original_format, destination_format, dir_worked, dir_images, name_image):    
    # Define los caminos de entrada y salida
    input_path = os.path.join(dir_worked, name_image + original_format)
    output_path = os.path.join(dir_images, name_image + destination_format)

    if not os.path.exists(output_path):
        # Lee la imagen
        image_rgba = Image.open(input_path)

        # Realiza cualquier otra manipulación necesaria (por ejemplo, recorte y redimensionamiento)
        
        # mask_image, _ = crop_fov(image_rgba)
        mask_image = crop_fov2(image_rgba)
        #mask_image = downsize_image(mask_image)

        # Convierte mask_image a un objeto PIL Image (si no lo es ya)
        if not isinstance(mask_image, Image.Image): # Nose que devuelve downsize_image
            mask_image = Image.fromarray(mask_image)

        # Convierte mask_image a RGB
        mask_image = mask_image.convert("RGB")

        # Guarda la imagen como archivo PNG
        mask_image.save(output_path, "PNG")



def downsize_image(fundus_picture, image_size=[512, 512]):
    '''
    Downsize the input image to the target resolution
    '''

    # get the proper size
    if fundus_picture.shape[0] <= fundus_picture.shape[1]:
        target_size = image_size[0] / fundus_picture.shape[0]
    else:
        target_size = image_size[0] / fundus_picture.shape[1]

    # apply the transformation
    resized_fundus_picture = np.asarray(rescale(fundus_picture, scale=target_size, multichannel=True, preserve_range=True), dtype=np.uint8)

    return resized_fundus_picture

def get_fov_mask(fundus_picture):
    '''
    Obtains the fov mask of the given fundus picture.
    '''
    
    # initialize an empty matrix for the FOV
    width, height = fundus_picture.size
    fov_mask = np.zeros((width, height), dtype=np.uint8)

    #fov_mask = np.zeros((fundus_picture.shape[0], fundus_picture.shape[1]), dtype=np.uint8)
    
    # estimate the center (x,y) and the radius (r) of the circle
    _, x, y, r = detect_xyr(fundus_picture)
    
    width, height = fundus_picture.size
    # generate a binary mask
    Y, X = np.ogrid[:width, :height]
    dist_from_center = np.sqrt((X - x)**2 + (Y-y)**2)
    fov_mask[dist_from_center <= r] = 255
                
    return fov_mask, (x,y,r)

def get_fov_mask2(fundus_picture):
    '''
    Obtains the fov mask of the given fundus picture.
    '''

    # sum the R, G, B channels to form a single image
    sum_of_channels = np.asarray(np.sum(fundus_picture,axis=2), dtype=np.uint8)
    # threshold the image using Otsu
    fov_mask = sum_of_channels > filters.threshold_otsu(sum_of_channels)
    # fill holes in the approximate FOV mask
    fov_mask = np.asarray(binary_fill_holes(fov_mask), dtype=np.uint8)

    return fov_mask

def crop_fov(fundus_picture, fov_mask=None):
    '''
    Crop an image around the fov, and return also the cropped fov
    '''
    fundus_picture = fundus_picture.convert('RGB')

    # if the FOV mask is not given, estimate it
    if fov_mask is None:
        fov_mask, (x, y, r) = get_fov_mask(fundus_picture)
    # if the FOV mask is given, estimate the center and the radii
    else:
        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)
        # Convertir la imagen PIL a un array de NumPy
    
    fundus_picture_np = np.array(fundus_picture)

    width, height = fundus_picture.size
    lim_x_inf = (y - r) if y>=r else 0
    lim_x_sup = (y + r) if (y+r)<width else width
    lim_y_inf = (x - r) if x>=r else 0
    lim_y_sup = (x + r) if (x+r)<height else height
    
    return fundus_picture_np[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup,:], fov_mask[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup]

def crop_fov2(fundus_picture):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    # Convertir la imagen a una matriz NumPy
    fundus_array = np.array(fundus_picture)  # Esto es nuevo!!!!

    #get the fov mask of the picture
    fov_mask = get_fov_mask2(fundus_picture)

    # get the coordinate of a bounding box around the fov mask
    coordinates = measure.regionprops(fov_mask)[0].bbox

    # crop the image and return
    #return fundus_picture[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3],:]
    return fundus_array[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3],:]



def detect_xyr(img):
    """
    Taken from https://github.com/linchundan88/Fundus-image-preprocessing/blob/master/fundus_preprocessing.py
    """
    
    # determine the minimum and maximum possible radii
    MIN_RADIUS_RATIO = 0.5
    MAX_RADIUS_RATIO = 1.0
    # get width and height of the image
    width, height = img.size
    # width = img.shape[1]
    # height = img.shape[0]

    # get the min length of the image
    myMinWidthHeight = min(width, height)
    # and get min and max radii according to the proportion
    myMinRadius = round(myMinWidthHeight * MIN_RADIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_RADIUS_RATIO)

    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # turn the image to grayscale
    #print("****** el tipo es: ", type(img))
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # estimate the circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, minDist=450, param1=200, param2=0.9,
                           minRadius=myMinRadius,
                           maxRadius=myMaxRadius)

    (x, y, r) = (0, 0, 0)
    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            x1, y1, r1 = circles[0]
            if x1 > (2 / 5 * width) and x1 < (3 / 5 * width) \
                    and y1 > (2 / 5 * height) and y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True

    # if the Hough transform couldn't get a circle, apply Nacho's technique
    if not found_circle:
        
        # sum the R, G, B channels to form a single image
        sum_of_channels = np.asarray(np.sum(img,axis=2), dtype=np.uint8)
        # threshold the image using Otsu
        fov_mask = sum_of_channels > filters.threshold_otsu(sum_of_channels)
        # fill holes in the approximate FOV mask
        fov_mask = np.asarray(binary_fill_holes(fov_mask), dtype=np.uint8)

        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    return found_circle, x, y, r


def save_split_file(splits_path, filename, dataset, training_names, validation_names, test_names, verbose=False):
    '''
    Save a training/validation/test split into a .ini file
    --------
    Inputs:
        splits_path : path to save the splits
        filename : name of the file
        dataset : name of the dataset
        training_names : list of files for the training set
        validation_names : list of files for the validation set
        test_names : list of files for the test set
        verbose : indicate whether you want to print the partition or not
    Outputs:
        split : file with the training/validation/test partition
    '''

    # correct the lists in case they have an empty value
    if '' in training_names:
        training_names.remove('')
    if '' in validation_names:
        validation_names.remove('')
    if '' in test_names:
        test_names.remove('')

    # export the split
    # - create the split file
    split = ConfigParser()
    split.add_section('split')
    split.set('split', 'type', 'holdout')
    split.set('split', 'training', list_to_string(training_names))
    split.set('split', 'validation', list_to_string(validation_names))
    split.set('split', 'test', list_to_string(test_names))
    # - save the split file
    split_file = open(path.join(splits_path, filename + '_' + dataset + '.ini'),'w')
    split.write(split_file)
    split_file.close()
    # - print the split
    print('- Training set: {} images'.format(len(training_names)))
    print('- Validation set: {} images'.format(len(validation_names)))
    print('- Test set: {} images'.format(len(test_names)))

    return split

    

def parse_boolean(input_string):
    '''
    Parse a boolean
    '''
    return input_string.upper()=='TRUE'



def list_to_string(input_list):
    '''
    Turn an input list of strings into a single string with comma separated values
    '''

    if (input_list == None) or (len(input_list)==0):
        return ''
    else:
        return ','.join(list(input_list))



def string_to_list(input_string):
    '''
    Turn a string with comma separated values into a list
    '''
    list_to_return = input_string.split(',')
    if (len(list_to_return) == 1) and (list_to_return[0]==''):
        list_to_return = []
    return list_to_return



def numpy_one_hot(input_array, num_classes):

    dims = [num_classes]
    for dim in list(input_array.shape):
        dims.append(dim)

    one_hot_encoding = np.zeros(tuple(dims))

    for class_id in range(num_classes):
        one_hot_encoding[class_id, :] = input_array==class_id

    return one_hot_encoding


def remove_extensions(filenames):
    '''
    Remove file extensions from filenames
    '''

    # initialize an empty list of files
    new_filenames = []
    # iterate for each filename 
    for i in range(len(filenames)):
        # get the filename and the extension
        filename_without_extension, file_extension = path.splitext(filenames[i])
        # append only the filename to the list
        new_filenames.append(filename_without_extension)
    
    return new_filenames



def natural_key(string_):
    '''
    To sort strings using natural ordering. See http://www.codinghorror.com/blog/archives/001018.html
    '''
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]




def get_predicted_and_ground_truth_overlap(predicted, groundtruth):
    '''
    Generates an RGB image with black in true negatives, red in false positives,
    green in true positives and blue in false negatives
    '''

    groundtruth = np.squeeze(groundtruth)

    green = np.logical_and(predicted, groundtruth)
    red = np.logical_and(predicted, np.logical_not(groundtruth))
    blue = np.logical_and(np.logical_not(predicted), groundtruth)

    rgb = np.array(np.stack((red, green, blue), axis=0) * 255.0, dtype=np.uint8)

    return rgb



def prepare_image_for_visdom_plotter(current_image):
    '''
    Adjust an input image so that it can be shown in a Visdom plotter
    '''

    # if there are more than 3 channels, then squeeze the image
    current_image = np.squeeze(current_image)

    # if the image don't have RGB channels
    if len(current_image.shape) < 3:
        # expand the RGB dimension
        current_image = np.expand_dims(current_image, axis=0)
    else:
        # transpose the image so that it has it in the right place
        current_image = np.transpose(current_image, [2, 0, 1])
    
    # if the RGB dimension has a single dimension
    if current_image.shape[0]==1:
        # repeat 3 times the same image in the RGB dimension
        current_image = np.concatenate((current_image, current_image, current_image), axis=0)

    if np.unique(current_image.flatten()).size == 2:
        current_image = current_image > 0
        current_image = np.array(current_image, dtype=np.uint8) * 255
    
    return current_image



def get_image_filenames_from_folder(input_path, include_path=True):
    '''
    Get a list of image filenames from folder
    '''

    # default file extensions
    extensions = ['.tif', '.png', '.bmp', '.gif', '.mat']
    
    # initialize an empty list of filenames
    image_filenames = []
    # iterate for each extension
    for ext in extensions:
        # collect the list of files with that extension
        current_files = glob(path.join(input_path, '*' + ext))
        # remove the path if necessary
        if not include_path:
            for i in range(len(current_files)):
                current_files[i] = path.basename(current_files[i])
        # concatenate
        image_filenames = image_filenames + current_files

    # sort image filenames
    image_filenames = sorted(image_filenames, key=natural_key) 

    return image_filenames




class SegmentationColorMapper(object):
    '''
    A class to map segmentations from colors and viceversa
    '''

    def __init__(self, config):
        '''
        Constructor
        '''

        # retrieve the list of colors from the config file
        self.classes_names = []
        self.classes_colors = []
        for current_class in config:
            self.classes_names.append(current_class)
            self.classes_colors.append(np.array(string_to_list(config[current_class]), dtype=np.uint8))

        # get the number of classes
        self.num_classes = len(self.classes_names)


    def map_colors_to_classes(self, labels):
        '''
        Map colors to classes
        '''

        # if no color channels are available...
        if len(labels.shape) < 3:

            # if only two classes are there
            if self.num_classes==2:
                
                # we generate a binary labelling
                labels = np.array(labels > 0, dtype=np.float32)
            
            # if more than 2 labels are there, we just leave them as they are
            else:
                
                # create an empty mask
                mask = np.zeros((labels.shape[0], labels.shape[1]))
                # iterate for each color and replace it by the label
                for i in range(self.num_classes):
                    mask[labels==self.classes_colors] = i
                # replace labels with the new mask
                labels = mask

        # if there are colors
        else:

            # create an empty mask
            mask = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)
            # iterate for each color and replace it by the label
            for i in range(self.num_classes):
                idx = (labels == np.array(self.classes_colors[i]))
                validx = (np.sum(idx, axis=2) == 3)
                mask[validx] = i
            # replace labels with the new mask
            labels = mask

        # return labels
        return labels


    def map_classes_to_colors(self, labels):
        '''
        Map classes to colors
        '''

        labels = np.squeeze(labels)

        red_channel = np.zeros(labels.shape)
        green_channel = np.zeros(labels.shape)
        blue_channel = np.zeros(labels.shape)

        # iterate for each class
        for i in range(self.num_classes):

            # replace RGB positions for this label with the right tones
            red_channel[labels==i] = self.classes_colors[i][0]
            green_channel[labels==i] = self.classes_colors[i][1]
            blue_channel[labels==i] = self.classes_colors[i][2]

        # reconstruct the color mapping
        color_coded_labels = np.array(np.stack((red_channel, green_channel, blue_channel), axis=2), dtype=np.uint8)

        return color_coded_labels

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', server='localhost', port='8097'):
        print("Visdom en:")
        print("url: ",server, "port: ", port)
        self.viz = Visdom('http://' + server, port = port)
        self.env = env_name
        self.plots = {}
    
    def image(self, img):
        self.viz.image(img)

    def image2(self, img, win):
        self.viz.image(img, win=win)
        
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
                #ytickmin=0,
                #ytickmax=1
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    
    def plot2(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name,
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count