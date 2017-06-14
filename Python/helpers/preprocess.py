import random
from os import walk
from os.path import join, basename, isfile, normpath

import helpers.labels as labels
import numpy as np
from PIL import Image


# ======================================================================================================
# Preprocess functions that can be used on images before feeding them into the net.
# Please refer to the description of every function
# ======================================================================================================


def produce_training_dir(imdir, labeldir, outdir, training_set_size, imW=640, imH=360, crop=True, alllabels=True):
    """
        Creates a folder containing cropped images.
        @ args :
            - imdir : directory of the training images
            - labeldir : directory of the label images
            - outdir : path to the folder where we want to write the new cropped images
            - training_set_size : the number of images to write
            - imW, imH : width and height of the cropping to perform
            - crop : whether to crop or not the images
        @ returns :
            - nothing, simply writes images
    """
    filelist = []
    imdir = normpath(imdir)
    labeldir = normpath(labeldir)
    for path, subdirs, files in walk(imdir):
        for name in files:
            splt_name = str(basename(name)).split(sep="_")
            img_name = join(path, name)
            city = splt_name[0]
            label_name = join(normpath(labeldir), city,
                              city + '_' + splt_name[1] + '_' + splt_name[2] + '_gtFine_labelIds.png')
            if (isfile(label_name)):
                filelist.append([img_name, label_name])
    out = []
    random_indices = np.random.randint(low=0, high=len(filelist), size=training_set_size)
    step = 0
    for i in random_indices:
        if (alllabels):
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imW))
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
        else:
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imW))
            Label = Image.eval(Label, labels.convert2trainId)
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
        print(step)
        step += 1
    return


def produce_training_dir_with_disp(imdir, labeldir, dispdir, outdir, training_set_size, imW=640, imH=360, crop=True,
                                   alllabels=True):
    """
        Creates a folder containing cropped images.
        @ args :
            - imdir : directory of the training images
            - labeldir : directory of the label images
            - outdir : path to the folder where we want to write the new cropped images
            - training_set_size : the number of images to write
            - imW, imH : width and height of the cropping to perform
            - crop : whether to crop or not the images
        @ returns :
            - nothing, simply writes images
    """
    filelist = []
    imdir = normpath(imdir)
    labeldir = normpath(labeldir)
    dispdir = normpath(dispdir)
    for path, subdirs, files in walk(imdir):
        for name in files:
            splt_name = str(basename(name)).split(sep="_")
            img_name = join(path, name)
            city = splt_name[0]
            label_name = join(normpath(labeldir), city,
                              city + '_' + splt_name[1] + '_' + splt_name[2] + '_gtFine_labelIds.png')
            disp_name = join(dispdir, city,
                             city + '_' + splt_name[1] + '_' + splt_name[2] + '_disparity.png')
            if (isfile(label_name) and isfile(disp_name)):
                filelist.append([img_name, label_name, disp_name])
    out = []
    random_indices = np.random.randint(low=0, high=len(filelist), size=training_set_size)
    step = 0
    for i in random_indices:
        if (alllabels):
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            Disp = Image.open(filelist[i][2])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
                Disp = Disp.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imH))
                Disp.thumbnail((imW, imH))
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
            Disp.save(join(outdir, '_' + str(step) + '_disp_.png'))
        else:
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            Disp = Image.open(filelist[i][2])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
                Disp = Disp.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imH))
                Disp.thumbnail((imW, imH))
            Label = Image.eval(Label, labels.convert2trainId)
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
            Disp.save(join(outdir, '_' + str(step) + '_disp_.png'))
        print(step)
        step += 1
    return


def produce_training_set(traindir, trainsize, numlabs=35):
    """
        Produces a list of training images and labels.
        @ args :
            - traindir : path to the directory containing training images.
            - trainsize : an integer, the size of the training set we want to use. Must be lower than the number of images in the folde
        @ returns :
            - ins : a 4D np array (trainsize* {imsize} * {channels}) of the images.
            - labs : a 4D np array (trainsize * {imsize} * {numlabs+1}) one-hot encoding labels. The last label (index {numlabs}) signifies unlabelled.
            - hist : list of histograms of the labels. 
    """

    num_labels = numlabs
    indices = list(range(trainsize))
    random.shuffle(indices)
    ins = []
    labs = []
    hist = np.zeros((num_labels))
    for i in indices:
        Im = Image.open(normpath(join(traindir, '_' + str(i) + '_im_.png')))
        im = np.asarray(Im, dtype=np.float32)
        Label = Image.open(join(traindir, '_' + str(i) + '_lab_.png'))
        lab = np.asarray(Label.convert(mode="L"), dtype=np.int)
        maxlabs = num_labels * np.ones_like(lab)
        lab = np.minimum(lab, maxlabs)
        new_hist, _ = np.histogram(lab, bins=num_labels + 1)
        lab = np.eye(num_labels + 1)[lab]
        # hist += new_hist
        ins.append(im)
        labs.append(lab)
    return np.asarray(ins), np.asarray(labs)  # , hist


def produce_training_set_with_disp(traindir, trainsize, numlabs=35):
    """
        Produces a list of training images and labels.
        @ args :
            - traindir : path to the directory containing training images.
            - trainsize : an integer, the size of the training set we want to use. Must be lower than the number of images in the folde
        @ returns :
            - ins : a 4D np array (trainsize* {imsize} * 4) of the images and disparity.
            - labs : a 4D np array (trainsize * {imsize} * {numlabs+1}) one-hot encoding labels. The last label (index {numlabs}) signifies unlabelled.
    """
    num_labels = numlabs
    indices = list(range(trainsize))
    random.shuffle(indices)
    ins = []
    labs = []
    hist = np.zeros((num_labels))
    for i in indices:
        Im = Image.open(normpath(join(traindir, '_' + str(i) + '_im_.png')))
        im = np.asarray(Im, dtype=np.float32)
        Label = Image.open(join(traindir, '_' + str(i) + '_lab_.png'))
        lab = np.asarray(Label.convert(mode="L"), dtype=np.int)
        Disp = Image.open(normpath(join(traindir, '_' + str(i) + '_disp_.png')))
        disp = np.asarray(Disp.convert(mode='L'), dtype=np.float32)
        disp = np.expand_dims(disp, axis=2)
        maxlabs = num_labels * np.ones_like(lab)
        lab = np.minimum(lab, maxlabs)
        new_hist, _ = np.histogram(lab, bins=num_labels + 1)
        lab = np.eye(num_labels + 1)[lab]
        ins.append(np.concatenate((im, disp), axis=-1))
        labs.append(lab)
    return (np.asarray(ins), np.asarray(labs))


def produce_testing_set(testdir, testsize=100, imH=128, imW=256):
    out = []
    for i in range(testsize):
        Im = Image.open(normpath(join(testdir, '_' + str(i) + '_im_.png')))
        Im.thumbnail([imW, imH])
        im = np.asarray(Im, dtype=np.float32)
        Label = Image.open(join(testdir, '_' + str(i) + '_lab_.png'))
        Label.thumbnail([imW, imH])
        lab = np.asarray(Label.convert(mode="L"), dtype=np.float32)
        out.append([im, lab])
    return out

def file_len(file):
    """
    Computes the number of lines in the file
    Args:
        file (string): path to the file

    Returns:
        int : number of line
    """
    i = 0
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i+1

def add_header(inputfile, outputfile):
    """
    Adds a header at the begining of the file
    Args:
        inputfile (string): path to the input file
        outputfile (string): path to the output file

    Returns:
        None
    """
    i = file_len(inputfile)
    infile = open(inputfile, 'r')
    outfile = open(outputfile, 'w')
    outfile.write("1\n")
    outfile.write(str(i)+"\n")
    header = ["0 0 0","1 0 0","0 1 0","0 0 1","1 0 0 0","0 1 0 0","0 0 1 0","0 0 0 1"]
    for line in header:
        outfile.write(line+"\n")
    for line in infile:
        outfile.write(line)
    outfile.close()
    infile.close()

def append_labels(pointsfile, labelsfile, outfile):
    """
    Just appends the labels at the end of each line
    Args:
        pointsfile (string): path to the file containting points 
        labelsfile (string): path to the file containing labels
        outfile (string): path to the file to be written

    Returns:
        None
    """
    with open(outfile,"w") as o:
        pf = open(pointsfile, "r")
        lf = open(labelsfile, "r")
        for point in pf:
            lab = lf.readline()
            o.write(point[:-1] + " " + lab)
        lf.close()
        pf.close()


#Dictionnary linking function names to the actual funcitons
set_builders = {
    'without_disp' : produce_training_set,
    'with_disp' : produce_training_set_with_disp
}