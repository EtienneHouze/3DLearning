"""
Petit script pour transformer les images étiquetées RGB en images ne contenant plus que les étiquettes.
Attribue la valeur 0 aux zones non étiquetées.
Syntaxe :
    <data_folder>
        data_folder : dossier contenant les répertoires de données, ainsi que la correspondance couleurs/etiquettes, conformément au readme.

Options :
    -h ou --help : affiche l'aide.
    --thresh <value> : Spécifie une proportion seuil de pixels non noirs dans l'image afin que celle ci soit gardée.
"""
# FIXME : pour le momennt, la valeur seuil fait planter la suite du programme car elle ne met pas a jour le fichier de projections des points sur les images...
# Fonction de M...
# TODO : prendre son courage a deux mains et essayer de parser le magnifique fichier ascii de plusieurs Go et lui enlever toutes les références aux images que l'on a supprimé... Yeay !

from __future__ import absolute_import, division, print_function

import fileinput
import getopt
import sys
from os import walk, mkdir, remove
from os.path import join, isdir

import numpy as np
from PIL import Image


def aux(a, labels):
    """
    Auxiliary function
    Args:
        a (np.array): a slice of a RGB image ( (3,) np array)

    Returns:

    """
    rep = np.zeros(shape=3, dtype='uint8')
    for i in range(len(labels)):
        if np.allclose(a, np.asarray(labels[i])[:-1], rtol=0, atol=5):
            rep[0] = labels[i][3]
    return rep


def main():
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ['help','thresh='])
    except getopt.error as msg:
        print(msg)
        print("try -h or --help")
        sys.exit(-1)

    thresh = 1

    for o, a in opts:
        if o in ['-h', '--help']:
            print(__doc__)
            sys.exit(2)
        if o in ['--thresh']:
            thresh = float(a)

    if len(args) != 1:
        print("try -h or --help")
        sys.exit(-1)

    data_root = args[0]
    rgb_folder = join(data_root,'RGB')
    rgb_lab_folder = join(data_root,'RGB_labels')
    lab_folder = join(data_root,'Labels')
    labels_file = join(data_root,'labels.txt')
    labels = []
    removed_files = []

    with fileinput.input((labels_file)) as lab_file:
        for line in lab_file:
            splt = line.split(sep=' ')
            if splt[0] == '#':
                pass
            else:
                label = [int(splt[i]) for i in range(4)]
                labels.append(label)

    if not isdir(lab_folder):
        mkdir(lab_folder)

    for root, dir, files in walk(rgb_folder):
        for file in files:
            if not ('.png' in file or '.jpg' in file):
                pass
            else:
                del_op = False
                if thresh < 1:
                    rgb_im = np.asarray(Image.open(join(rgb_folder,file)))
                    rgb_im = np.cumsum(rgb_im, axis = -1)
                    num_zeros = np.count_nonzero(rgb_im)
                    if num_zeros > thresh * rgb_im.size:
                        del_op = True
                if not del_op:
                    im = Image.open(join(rgb_lab_folder, file))
                    im_array = np.asarray(im)
                    lab_array = np.apply_along_axis(lambda a: aux(a, labels), axis=-1, arr=im_array)
                    lab = Image.fromarray(lab_array[:, :, 0])
                    lab.save(join(lab_folder, file))
                    print("Image " + file + " processed.")
                else :
                    removed_files.append(file)
                    remove(join(rgb_folder,file))
                    remove(join(data_root,'Depth',file))
                    remove(join(data_root,'Altitude',file))
                    remove(join(rgb_lab_folder,file))
                    print("Image "+str(file)+" ramoved because not relevant enough.")
    # with fileinput.input((join(data_root,'Projections.txt'))) as f:


if __name__ == "__main__":
    main()
