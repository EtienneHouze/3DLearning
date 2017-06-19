"""
Script se chargeant de convertir les images d'un niveau de gris vers les couleurs correspondant aux labels
Syntaxe :
    <im_folder> <labels_file> [<out_folder>]
        im_folder : dossier contenant les images
        labels_file : fichier de correspondance label/RGB
        [optionnal] out_folder : dossier ou enregistrer les images
"""

from __future__ import absolute_import, print_function, division
import getopt
import sys
import fileinput
import numpy as np
from os.path import join, isfile, dirname, basename, isdir
from os import listdir, mkdir
from PIL import Image

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



def main():
    try:
        opt,args = getopt.gnu_getopt(sys.argv[1:],'',[])
    except getopt.error as msg:
        print(msg)
        print(__doc__)
        sys.exit(-1)
    im_folder = args[0]
    if len(args)==3:
        out_folder = args[2]
        if not isdir(out_folder):
            mkdir(out_folder)
    else:
        out_folder = None
    label_file = args[1]
    labels_dict = {0:[0, 0, 0]}
    with fileinput.input((label_file)) as f:
        for line in f:
            splt = line.split(sep=' ')
            if splt[0]=='#':
                pass
            else:
                labels_dict[int(splt[3])] = [int(splt[0]),int(splt[1]),int(splt[2])]
    image_list = []
    for im in listdir(im_folder):
        if not 'processed' in im:
            image_list.append(join(im_folder,im))
    for image in image_list:
        if (isfile(image)):
            name_dir = dirname(image)
            im_name = basename(image)
            if out_folder:
                out_name = join(out_folder,im_name)
            else:
                out_name = join(name_dir,im_name + '_processed.png')
            Im = Image.open(image)
            im = np.asarray(Im, dtype=int)
            new_im = np.expand_dims(im, axis=2)
            new_im = np.tile(new_im, [1, 1, 3])
            new_im = np.apply_along_axis(lambda a:lab2color(a,labels_dict),axis=-1,arr=new_im)
            Out = Image.fromarray(new_im.astype('uint8'))
            Out.save(out_name)


if __name__=="__main__":
    main()