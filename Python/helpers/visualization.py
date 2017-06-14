from __future__ import absolute_import, division, print_function

import csv
from os import walk
from os.path import join, basename, dirname, isfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def moving_average(x, size):
    """
    helper function to compute the moving average
    :param x: input array
    :type x: np.ndarray
    :param size: size of the moving averaging window
    :type size: integer
    :return: averaged array
    :rtype: np.ndarray
    """
    window = np.ones(int(max(1, size))) / float(max(1, size))
    return np.convolve(x, window, 'same')


def visualize_csvlog(filepath, **kwargs):
    """
    A function to vizualize  logs as registered in the logs subfolder of models
    :param filepath: the path to the file
    :param metrics (optional): for now, either "loss" or "IoU"
    :param mode (optional): for now, either 'train' or 'val'
    :param 'scale' (optional): either 'linear' or 'log'
    :param smoothing (optional): smoothing factor for the data points
    :return: 
    """
    label_x = 'iterations'
    label_y = ''
    title = ''
    scale = 'linear'
    smoothing = 0
    for key in kwargs.keys():
        if key == 'metrics':
            mode = kwargs.get(key)
            if mode == 'loss':
                label_y = 'Loss'
            if mode == 'IoU' or mode == 'iou':
                label_y == 'IoU'
        if key == 'mode':
            mode = kwargs.get(key)
            if mode == 'train':
                title = 'Training'
            if mode == 'val':
                title = 'Validation'
        if key == 'scale':
            scale = kwargs.get(key)
        if key == 'smoothing':
            smoothing = kwargs.get(key)
    data = [[], []]
    with open(filepath, 'rt') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row:
                print(row)
                data[0].append(float(row[0]))
                data[1].append(float(row[1]))
            print(type(row))
        y_data = moving_average(data[1], len(data[1]) * smoothing)
        x_data = data[0]
        plt.plot(x_data, y_data)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.yscale(scale)
        plt.title(title)
        plt.show()
        print('done')


def visualize_csv(filepath, unique_graph=True, scales={}, means={}):
    """
    
    Args:
        filepath (string): path to the csv file 
        unique_graph (bool): whether to print a single graph or not 
        scales (dict): specifies whether to use particular scales for metrics 
        means (dict): specifies the size of the window for averaging

    Returns:
        Nothing
    """
    matplotlib.style.use('ggplot')
    data = pd.read_csv(filepath)
    for key in data.keys():
        data[key] = pd.to_numeric(data[key],errors='coerce')
    data = data.dropna()
    for key in means.keys():
        if key in data.keys():
            data[key] = data[key].rolling(window=means.get(key),center=False).mean()
    if unique_graph and scales == {}:
        data.plot()
    else:
        if scales != {}:
            for key in scales.keys():
                if key in data.keys():
                    plt.figure()
                    plt.ylabel(key)
                    plt.xlabel('iterations')
                    plt.title(key)
                    plt.yscale(scales.get(key))
                    data[key].plot()
        else:
            for key in data.keys():
                plt.figure()
                plt.ylabel(key)
                plt.xlabel('iterations')
                plt.title(key)
                data[key].plot()
    plt.show()


def lab2color(lab, colors={}):
    """
    A simple helper function which computes a color, given a label
    :param lab: the input label
    :type lab: a 1D, one-elemenent np array
    :return: color, the RGB color of this label
    :rtype: A 1D, 3-element np array
    """
    if lab[0] in colors.keys():
        return np.asarray(colors.get(lab[0]))
    else:
        return np.zeros((3))


def convert_labelled_images(image_list=[], num_labs=18):
    """
    Method to visualize easily labelled images by attributing
    :param image_list: paths to the images to process
    :param num_labs: number of classes
    :type num_labs: integer
    :type image_list: an iterable of strings
    :return: nothing, writes images in the same folder as the input images.
    :rtype: None
    """
    colors_dict = {}
    for i in range(num_labs):
        colors_dict[i] = np.random.randint(low=0, high=256, size=3)
    for image in image_list:
        if (isfile(image)):
            name_dir = dirname(image)
            im_name = basename(image)
            out_name = im_name + '_processed.png'
            Im = Image.open(image)
            im = np.asarray(Im, dtype=int)
            new_im = np.expand_dims(im, axis=2)
            new_im = np.tile(new_im, [1, 1, 3])
            it = np.nditer(im, flags=['multi_index'])
            while not it.finished:
                new_im[it.multi_index] = lab2color(new_im[it.multi_index], colors_dict)
                it.iternext()
            Out = Image.fromarray(new_im.astype('uint8'))
            Out.save(join(name_dir, out_name))


#
def convert_labelled_output(dir_name, num_labs=18):
    """
    Convert every image in 
    :param dir_name: directory containing the labelled images
    :type dir_name: string
    :param num_labs: number of classes
    :type num_labs: integer
    :return: nothing
    :rtype: None
    """
    imlist = []
    for root, _, files in walk(dir_name):
        for name in files:
            if 'output' in name:
                imlist.append(join(root, name))
    convert_labelled_images(imlist, num_labs)
