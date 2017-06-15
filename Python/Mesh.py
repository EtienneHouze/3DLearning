from __future__ import absolute_import,print_function,division

from os.path import isfile, join, dirname
from Point import Point
import fileinput
import sys
import numpy as np
from PIL import Image
from statistics import mode

class Mesh:

    def __init__(self, obj):
        self.verts = []
        self.numpoints = 0
        self.labels = []
        if not isfile(obj):
            pass
        self.root = dirname(obj)
        with fileinput.input((obj),mode='r') as file:
            for line in file:
                if len(line)==0:
                    break
                splt = line.split(sep=' ')
                if splt[0] == 'v':
                    self.verts.append(Point(*splt[1:]))
                elif splt[0] == 'vt':
                    break
        self.numpoints = len(self.verts)

    def load_labels(self, label_file):
        """
        Charge les étiquettes depuis un fichier annexe
        Args:
            label_file (string): chemin vers le fichier annexe

        Returns:
            Complète le champ labels du mesh
        """
        if not isfile(label_file):
            print("label file not found")
            sys.exit(-1)
        else:
            with fileinput.input((label_file)) as file:
                for line in file:
                    splt = line.split(sep=' ')
                    if len(splt) != 4:
                        print("wrong line")
                    else:
                        line_list = []
                        for s in splt:
                            line_list.append(int(s))
                        self.labels.append(line_list)


    def labelise(self,projection_file,lab_dir):
        # TODO : essayer de voir comment faire cette éthode raisonnablement bien...
        """
        Méthode qui attribue a chaque point du mesh son étiquette, selon le jeu d'images dont les projections sont définies dans projection_file
        Args:
            projection_file (string): fichier de sortie du Projection.exe de Victor.

        Returns:
            Met à jour les labels des points du Mesh.
        """
        if not isfile(projection_file):
            pass
        num_textures = 0
        images = []
        with fileinput.input((projection_file),mode='r') as proj:
            point_index = 0
            for line in proj:
                if len(line)==0:
                    break
                splt = line.split(sep=' ')
                if len(splt)==1:
                    num_textures = int(splt[0])
                elif len(splt) == 2:
                    im = Image.open(join(lab_dir,splt[0]))
                    images.append(np.asarray(im,dtype='uint8'))
                    print("image loaded")
                else:
                    # TODO
                    labels = []
                    num_images = len(splt)//3
                    for i in range(num_images):
                        im_index = int(splt[3*i])
                        x_coord = np.floor(float(splt[3*i+1]))
                        y_coord = np.floor(float(splt[3*i+2]))
                        labels.append(images[im_index][x_coord,y_coord])
                    lab = mode(labels)
                    self.verts[point_index].label = lab
                    point_index += 1


