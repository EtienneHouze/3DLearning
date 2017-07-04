from __future__ import absolute_import, print_function, division

import fileinput
import sys
from os.path import isfile, join, dirname

import numpy as np
from PIL import Image

from Point import Point


# TODO : tester cette classe pour voir si les méthodes marchent bien comme il faut...
class Mesh:
    """
    Une classe définissant un mesh, décrit simplement comme un nuage de points.
    """

    def __init__(self, obj):
        """
        Initialise le mesh en chargeant le nuage de points depuis le fichier .obj spécifié.
        Args:
            obj (string): fichier .projections.txt décrivant le mesh
        """
        self.verts = []
        self.numpoints = 0
        self.labels = []
        if not isfile(obj):
            pass
        self.root = dirname(obj)
        with fileinput.input((obj), mode='r') as file:
            for line in file:
                if len(line) == 0:
                    break
                # splt = line.split(sep=' ')
                # if splt[0] == 'v':
                #     self.verts.append(Point(*splt[1:]))
                # elif splt[0] == 'vt':
                #     break
                splt = line.split(' ')
                if len(splt) < 3:
                    pass
                else:
                    self.verts.append(Point(*splt[0:3]))
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

    def labelise(self, projection_file, lab_dir, min_thresh=0):
        """
        Méthode qui attribue a chaque point du mesh son étiquette, selon le jeu d'images dont les projections sont définies dans projection_file
        Args:
            projection_file (string): fichier de sortie du Projection.exe de Victor.
            lab_dir (string) : dossier contenant les étiquettes du modèle, en niveaux de gris...

        Returns:
            Met à jour les labels des points du Mesh.
        """
        if not isfile(projection_file):
            pass
        images = []
        with fileinput.input((projection_file), mode='r') as proj:
            point_index = 0
            for line in proj:
                if len(line) == 0:
                    break
                splt = line.split(sep=' ')
                if len(splt) == 1:
                    pass
                elif len(splt) == 2:
                    im = Image.open(join(lab_dir, splt[0].replace('JPG', 'png')))
                    images.append(np.asarray(im, dtype='uint8'))
                    print("image loaded")
                else:
                    labels = []
                    splt = splt[3:]
                    num_images = len(splt) // 3
                    for i in range(num_images):
                        im_index = int(splt[3 * i])
                        x_coord = int(np.floor(float(splt[3 * i + 1])))
                        y_coord = int(np.floor(float(splt[3 * i + 2])))
                        labels.append(images[im_index][y_coord, x_coord])
                    lab = max(set(labels), key=lambda e: labscore(e, labels, min_thresh))
                    self.verts[point_index].label = lab
                    print("point " + str(point_index) + " processed")
                    point_index += 1

    def save_to_txt(self, file):
        with open(file, mode='w') as f:
            f.write(str(self.numpoints) + '\n')
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

    def save_to_ptx(self, file, label_file, write_labels=False):
        """
        Exporte le mesh au format ptx avec les
        Args:
            file (string): nom du fichier à écrire
            label_file (string): fichier donnant la correspondance étiquettes/couleurs
            write_labels (bool) : Si l'on écrit les étiquette dans le ptx ou non

        Returns:

        """
        labels_dict = {0: [0, 0, 0]}
        with fileinput.input((label_file)) as f:
            for line in f:
                splt = line.split(sep=' ')
                if splt[0] == '#':
                    pass
                else:
                    r = int(splt[0])
                    g = int(splt[1])
                    b = int(splt[2])
                    l = int(splt[3])
                    labels_dict[l] = [r, g, b]

        with open(file, mode='w') as f:
            header = "1" + "\n" + str(len(
                self.verts)) + "\n" + "0 0 0\n" + "1 0 0\n" + "0 1 0\n" + "0 0 1\n" + "1 0 0 0\n" + "0 1 0 0\n" + "0 0 1 0\n" + "0 0 0 1\n"
            f.write(header)
            for vert in self.verts:
                if write_labels:
                    line = str(vert.x) + " " + str(vert.y) + " " + str(vert.z) + " 0 "
                    for i in labels_dict[vert.label]:
                        line += (str(i) + " ")
                    line += (str(vert.label) + "\n")
                else:
                    line = str(vert.x) + " " + str(vert.y) + " " + str(vert.z) + " 0"
                    for i in labels_dict[vert.label]:
                        line += (" " + str(i))
                    line += "\n"
                f.write(line)


def labscore(e, l, min):
    if e == 0 and l.count(e) > min * len(l):
        return len(l)
    elif e == 0:
        return 0
    else:
        return l.count(e)


def Mesh_IoU(mesh_true, mesh_pred):
    """
    Calcule la distance IoU entre deux meshes étiquetés.
    Args:
        mesh_true (Mesh): mesh étiquté avec les labels exacts
        mesh_pred (Mesh): mesh étiquetés avec les prédictions

    Returns:
        La mesure IoU entre les deux Meshes.
    """

#     TODO : écrire cette méthode...
