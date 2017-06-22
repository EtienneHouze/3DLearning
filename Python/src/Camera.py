from __future__ import absolute_import, print_function,division

import xml.etree.ElementTree as ET
import numpy as np

class Camera:
    """
    Classe décrivant une caméra
    """

    def __init__(self, id = 0):
        self.id = id
        self.image_path = 'R:/Data/Nashville/2014-11-30_interchange_img/IMG_'+str(self.id)+'.JPG'
        self.M_00 = 0
        self.M_01 = 0
        self.M_02 = 0
        self.M_10 = 0
        self.M_11 = 0
        self.M_12 = 0
        self.M_20 = 0
        self.M_21 = 0
        self.M_22 = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.lat = 0
        self.long = 0
        self.alt = 0
        self.focal = 0
        self.near = 0
        self.med = 0
        self.far = 0


    def parse_from_xml(self, xml_string):
        """

        Args:
            xml_string (string):

        Returns:
            complete the fields from the parsed string
        """
        pass

    def to_Element(self):
        """
        Fonction qui convertit la caméra en élément afin d'être ajoutée à un fichier xml.
        Returns:
            photo (ET.Element) : correspond au bloc <Photo> du fichier xml.
        """
        photo = ET.Element('Photo')
        id = ET.SubElement(photo,'Id')
        id.text = str(self.id)
        im_path = ET.SubElement(photo,'ImagePath')
        im_path.text = self.image_path
        comp = ET.SubElement(photo,'Component')
        comp.text = '1'
        pose = ET.SubElement(photo,'Pose')
        rot = ET.SubElement(pose,'Rotation')
        m00 = ET.SubElement(rot,'M_00')
        m01 = ET.SubElement(rot,'M_01')
        m02 = ET.SubElement(rot,'M_02')
        m10 = ET.SubElement(rot,'M_10')
        m11 = ET.SubElement(rot,'M_11')
        m12 = ET.SubElement(rot,'M_12')
        m20 = ET.SubElement(rot,'M_20')
        m21 = ET.SubElement(rot,'M_21')
        m22 = ET.SubElement(rot,'M_22')
        m00.text = str(self.M_00)
        m01.text = str(self.M_01)
        m02.text = str(self.M_02)
        m10.text = str(self.M_10)
        m11.text = str(self.M_11)
        m12.text = str(self.M_12)
        m20.text = str(self.M_20)
        m21.text = str(self.M_21)
        m22.text = str(self.M_22)
        center = ET.SubElement(pose,'Center')
        x = ET.SubElement(center,'x')
        y = ET.SubElement(center,'y')
        z = ET.SubElement(center,'z')
        x.text = str(self.x)
        y.text = str(self.y)
        z.text = str(self.z)
        near = ET.SubElement(photo,'NearDepth')
        near.text = str(self.near)
        med = ET.SubElement(photo,'MediumDepth')
        med.text = str(self.med)
        far = ET.SubElement(photo,'FarDepth')
        far.text = str(self.far)
        exif = ET.SubElement(photo,'ExifData')
        gps = ET.SubElement(exif,'GPS')
        lat = ET.SubElement(gps,'Latitude')
        long = ET.SubElement(gps,'Longitude')
        alt = ET.SubElement(gps,'Altitude')
        lat.text = str(self.lat)
        long.text = str(self.long)
        alt.text = str(self.alt)
        focal = ET.SubElement(exif,'FocalLength')
        focal.text = str(self.focal)
        maker = ET.SubElement(exif,'Make')
        maker.text = 'Canon'
        model = ET.SubElement(exif,'Model')
        model.text = 'random'
        date = ET.SubElement(exif,'DateTimeOriginal')
        date.text = '2014-11-30T20:29:16'

        indent(photo)
        return photo

    def generate_from_bounding_position(self, x=0,y=0,z=0,theta=[],phi=[],psi=[], focal=[]):
        """
        Met à jour la pose de la caméra afin que les coordonnées correspondent aux inputs, et que les angles de rotations soient dans les intervalles donnés.
        Args:
            x (float): position x
            y (float):
            z (float):
            theta (interval):
            phi (interval):
            psi (interval):
            focal (interval): précise l'intervalle dans lequel se trouve la focale de la caméra

        Returns:
            Met a jour l'objet caméra
        """
        if x:
            self.x = x
        if y:
            self.y = y
        if z:
            self.z = z
        if theta:
            rand_theta = float(np.random.uniform(theta[0],theta[1],1))
        else:
            rand_theta = 0
        if phi:
            rand_phi = float(np.random.uniform(phi[0],phi[1],1))
        else :
            rand_phi = 0
        if psi:
            rand_psi = float(np.random.uniform(psi[0],psi[1],1))
        else :
            rand_psi = 0
        if focal:
            self.focal = float(np.random.uniform(focal[0],focal[1],1))
        self.M_00 = np.cos(rand_psi)*np.cos(rand_phi) - np.sin(rand_psi)*np.cos(rand_theta)*np.sin(rand_phi)
        self.M_10 = np.cos(rand_psi)*np.cos(rand_phi) + np.sin(rand_psi)*np.cos(rand_theta)*np.sin(rand_phi)
        self.M_20 = np.sin(rand_theta)*np.cos(rand_phi)
        self.M_01 = -np.cos(rand_psi)*np.sin(rand_phi) - np.sin(rand_psi)*np.cos(rand_theta)*np.cos(rand_phi)
        self.M_11 = -np.cos(rand_psi)*np.sin(rand_phi) + np.sin(rand_psi)*np.cos(rand_theta)*np.cos(rand_phi)
        self.M_21 = np.sin(rand_theta) * np.cos(rand_phi)
        self.M_02 = np.sin(rand_psi)*np.sin(rand_theta)
        self.M_12 = - np.cos(rand_psi) * np.sin(rand_theta)
        self.M_22 = np.cos(rand_theta)

    def assign_focus_point(self,target_x, target_y, target_z, phi=[]):
        """
        Met à jour la rotation de la caméra pour qu'elle puisse viser un point donné
        Args:
            target_x (float): coordonnée x de la cible
            target_y (float): idem pour y
            target_z (float): idem pour z
            phi (interval) : interval de rotations possibles autour de l'axe de visée de la caméra

        Returns:
            Mets à jour la matrice de rotaions, ainsi que la profondeur.
        """
        if phi :
            phi = float(np.random.uniform(phi[0],phi[1],1))
        else :
            phi = 0
        vector = np.array((target_x-self.x,target_y-self.y,target_z-self.z))
        self.med = float(np.linalg.norm(vector))
        self.near = self.med - np.sqrt(self.med)
        self.far = self.med + np.sqrt(self.med)
        # Définition des vecteurs de la base liée à la caméra
        w = vector/self.med
        u = np.array((-float(w[1]),float(w[0]),0))
        u = u / np.linalg.norm(u)
        v = np.cross(w,u)
        M = np.array((u,v,w))
        M = np.transpose(M)
        # Matrice de roation d'angle phi
        c = np.cos(phi)
        s = np.sin(phi)
        R = np.zeros(shape=(3,3))
        R[0,0] = c
        R[0,1] = -s
        R[0,2] = 0
        R[1,0] = s
        R[1,1] = c
        R[1,2] = 0
        R[2,0] = 0
        R[2,1] = 0
        R[2,2] = 1
        # La matrice de rotation finale est obtenue par produit des deux rotations
        M = np.dot(M,R)
        self.M_00 = float(M[0,0])
        self.M_01 = float(M[0,1])
        self.M_02 = float(M[0,2])
        self.M_10 = float(M[1,0])
        self.M_11 = float(M[1,1])
        self.M_12 = float(M[1,2])
        self.M_20 = float(M[2,0])
        self.M_21 = float(M[2,1])
        self.M_22 = float(M[2,2])

def indent(elem, level=0):
    """
    Helper qui permet de rajouter des sauts de ligne.
    Args:
        elem (ET.Element): élément auquel on veut ajouter des sauts de ligne
        level (int): le niveau jusqu'auquel on ajoute des sauts de ligne

    Returns:
        Modifie elem pour y ajouter des sauts de ligne.
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i