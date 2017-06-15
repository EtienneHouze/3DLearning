"""
Petit script pour transformer les images étiquetées RGB en images ne contenant plus que les étiquettes.
Attribue la valeur 0 aux zones non étiquetées.
Synthaxe : <in_folder> <out_folder> <labels>
        in_folder : dossier contentant les images.
        out_folder : dossier de sortie.
        labels : fichier contenant les correspondances couleur->étiquettes

Options :
    -h ou --help : affiche l'aide.
"""
from __future__ import absolute_import, division, print_function

import fileinput
import getopt
import sys
from os import walk, mkdir
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
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ['help'])
    except getopt.error as msg:
        print(msg)
        print("try -h or --help")
        sys.exit(-1)

    for o, a in opts:
        if o in ['-h', '--help']:
            print(__doc__)
            sys.exit(2)

    if len(args) != 3:
        print("try -h or --help")
        sys.exit(-1)


    in_folder = args[0]
    out_folder = args[1]
    labels_file = args[2]
    labels = []

    with fileinput.input((labels_file)) as lab_file:
        for line in lab_file:
            splt = line.split(sep=' ')
            if splt[0] == '#':
                pass
            else:
                label = [int(splt[i]) for i in range(4)]
                labels.append(label)

    if not isdir(out_folder):
        mkdir(out_folder)

    for root, dir, files in walk(in_folder):
        for file in files:
            if not ('.png' in file or '.jpg' in file):
                pass
            else:
                im = Image.open(join(root, file))
                im_array = np.asarray(im)
                lab_array = np.apply_along_axis(lambda a: aux(a, labels), axis=-1, arr=im_array)
                lab = Image.fromarray(lab_array[:, :, 0])
                lab.save(join(out_folder, file))
                print("Image " + file + " processed.")


if __name__ == "__main__":
    main()
