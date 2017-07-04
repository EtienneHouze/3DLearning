"""
Script permettant de réaliser un inférence directe à partir d'un modele sur des images d'input.
Syntaxe :
    <model_folder> <im_folder>
        model_folder : dossier contenant le modele
        im_folder : dossier contenant les images a traiter
Options:
    -h ou --help : affiche l'aide
    -f ou --folder <out_dir> : sélectionne le dossier indiquer pour savegader les sorties.
"""
from __future__ import absolute_import, print_function, division

import getopt
import sys
import numpy as np
from PIL import Image
from os import walk
from os.path import join
from helpers.preprocess import dir_size
from src.CityScapeModel import CityScapeModel


def main():
    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'hf:',['help','folder='])
    except getopt.error as msg:
        print(msg)
        sys.exit(-1)

    model_folder = args[0]
    root_folder = args[1]
    out_dir = None
    model = CityScapeModel(model_folder)
    model.compile()
    model.load_weights()

    for o,a in opts:
        if o in ['-h','--help']:
            print(__doc__)
            sys.exit(0)
        if o in ['-f','--folder']:
            out_dir = a
    counter = 0
    im_folder = join(root_folder,'RGB')
    depth_folder = join(root_folder,'Depth')
    alt_folder = join(root_folder,'Altitude')
    for root,dirs,files in walk(im_folder):
        for file in files:
            if '.png' or 'jpg' in file:
                im_rgb = Image.open(join(im_folder,file))
                im_alt = Image.open(join(alt_folder,file))
                im_depth = Image.open(join(alt_folder,file))

                arr_rgb = np.asarray(im_rgb)
                arr_alt = np.asarray(im_alt)
                arr_alt = np.expand_dims(arr_alt,-1)
                arr_depth = np.asarray(im_depth)
                arr_depth = np.expand_dims(arr_depth,-1)
                x = np.concatenate((arr_rgb,arr_depth,arr_alt), axis=-1)
                x = np.squeeze(x)
                y = model.compute_output(x)
                y = np.squeeze(y)
                out = Image.fromarray(y.astype('uint8'))
                if out_dir:
                    out.save(join(out_dir,file))
                else:
                    out.save('out'+file)
                print('Imgae ' +str(counter) + 'processed')
                counter += 1

if __name__=="__main__":
    main()