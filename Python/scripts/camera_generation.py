"""
Script permettant de lancer la génération de caméras aléatoires
Syntaxe :
    <xml file> <num_cam>
        xml_file : fichier d'export xml
        num_cams : nombre de caméras à générer.
        first_id : id de la premiere image
Options :
    -h : affiche l'aide
    -x <min,max> : précise un intervalle en x.
    -y <min,max> : précise un intervalle en y.
    -z <min,max> : même chose en z.
    -phi <min,max>
    -theta <min,max>
    -psi <min,max>
    -f <min,max> : pour la focale
    --target_point <target_x_min,target_x_max;target_y_min,target_y_max;target_z_min,target_z_max> : point a centrer dans la caméra.
    --relative_target <x_range,y_range> : Cette option est exclusive avec l'option précédente. Elle permet de donner un point cible à la caméra, compris situé à plus ou moins (x_range, y_range) de la position (x,y) de la caméra.
"""
from __future__ import absolute_import, print_function, division

import getopt
import sys
import xml.etree.ElementTree as ET
from src.Camera import Camera
import numpy as np

def main():

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'hx:y:z:f:',['phi=','psi=','theta=','target_point=','relative_target='])
    except getopt.error as msg:
        print(msg)
        print(__doc__)
        sys.exit(-1)

    x = []
    y = []
    z = []
    theta = []
    psi = []
    phi = []
    focal = []
    target = False


    xml_file = args[0]
    num_cam = int(args[1])
    first_id = int(args[2])
    tree = ET.ElementTree(file=xml_file)
    root = tree.getroot()
    block = root.find('Block')
    groups = block.find('Photogroups')
    group = groups.find('Photogroup')
    for i in range(num_cam):
        for o, a in opts:
            if o == '-h':
                print(__doc__)
                sys.exit(1)
            if o == '-x':
                a = a.split(',')
                x = float(np.random.uniform(float(a[0]), float(a[1]), 1))
            if o == '-y':
                a = a.split(',')
                y = float(np.random.uniform(float(a[0]), float(a[1]), 1))
            if o == '-z':
                a = a.split(',')
                z = float(np.random.uniform(float(a[0]), float(a[1]), 1))
            if o == '--phi':
                a = a.split(',')
                phi = [float(a[0]), float(a[1])]
            if o == '--psi':
                a = a.split(',')
                psi = [float(a[0]), float(a[1])]
            if o == '--theta':
                a = a.split(',')
                theta = [float(a[0]), float(a[1])]
            if o == '-f':
                a = a.split(',')
                focal = [float(a[0]), float(a[1])]
            if o == '--target_point':
                a = a.split(sep=',')
                target_x = float(np.random.uniform(float(a[0]), float(a[1]), 1))
                target_y = float(np.random.uniform(float(a[2]), float(a[3]), 1))
                target_z = float(np.random.uniform(float(a[4]), float(a[5]), 1))
                target = True
            if o == '--relative_target' and not target:
                a = a.split(',')
                target_x = float(np.random.uniform(x - float(a[0]), x + float(a[0]), 1))
                target_y = float(np.random.uniform(y - float(a[1]), y + float(a[1]), 1))
                target_z = 0
                target = True
        cam = Camera(first_id)
        cam.generate_from_bounding_position(x=x,
                                            y=y,
                                            z=z,
                                            theta=theta,
                                            psi=psi,
                                            phi=phi,
                                            focal=focal
                                            )
        if target:
            cam.assign_focus_point(target_x=target_x,
                                   target_y=target_y,
                                   target_z=target_z,
                                   phi=phi)
            target = False
        cam_elem = cam.to_Element()
        group.append(cam_elem)
        print(ET.tostring(cam_elem))
        first_id += 1
    tree.write(xml_file,xml_declaration=True,encoding='utf-8',short_empty_elements=False)

if __name__ == "__main__":
    main()