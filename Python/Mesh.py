from __future__ import absolute_import,print_function,division

from os.path import isfile, join, dirname
from Point import Point
import fileinput
import sys
import numpy as np
from PIL import Image
from statistics import mode
# TODO : tester cette classe pour voir si les méthodes marchent bien comme il faut...
class Mesh:
    """
    Une classe définissant un mesh, décrit simplement comme un nuage de points.
    """

    def __init__(self, obj):
        """
        Initialise le mesh en chargeant le nuage de points depuis le fichier .obj spécifié.
        Args:
            obj (string): fichier .obj décrivant le mesh
        """
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
        """
        Méthode qui attribue a chaque point du mesh son étiquette, selon le jeu d'images dont les projections sont définies dans projection_file
        Args:
            projection_file (string): fichier de sortie du Projection.exe de Victor.

        Returns:
            Met à jour les labels des points du Mesh.
        """
        if not isfile(projection_file):
            pass
        images = []
        with fileinput.input((projection_file),mode='r') as proj:
            point_index = 0
            for line in proj:
                if len(line)==0:
                    break
                splt = line.split(sep=' ')
                if len(splt)==1:
                    pass
                elif len(splt) == 2:
                    im = Image.open(join(lab_dir,splt[0].replace('JPG','png')))
                    images.append(np.asarray(im,dtype='uint8'))
                    print("image loaded")
                else:
                    labels = []
                    num_images = len(splt)//3
                    for i in range(num_images):
                        im_index = int(splt[3*i])
                        x_coord = int(np.floor(float(splt[3*i + 1])))
                        y_coord = int(np.floor(float(splt[3*i + 2])))
                        labels.append(images[im_index][x_coord,y_coord])
                    lab = max(set(labels), key=labels.count)
                    self.verts[point_index].label = lab
                    print("point "+str(point_index) + " processed")
                    point_index += 1

    def save_to_txt(self, file):
        with open(file,mode='w') as f:
            f.write(str(self.numpoints)+'\n')
            for point in self.verts:
                f.write(str(point))
                f.write('\n')

    def load_from_txt(self, file):
        """
        Loads a mesh from a txt file
        Args:
            file (string): txt file

        Returns:
            Met à jour le champ verts du mesh.
        """
        with fileinput.input((file)) as f:
            for line in f:
                splt = line.split(sep=' ')
                if len(splt) != 4:
                    pass
                else:
                    self.verts.append(Point(*splt))
            self.numpoints = len(self.verts)