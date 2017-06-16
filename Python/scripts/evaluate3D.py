"""
Script chargeant et appliquant un modèle sur un set d'évaluation, en enregistrant les métriques et pouvant afficher les images produites.
Syntaxe :
    <model_folder> <validation_folder> <output_folder>
        model_folder : chemin vers le dossier contenant le modele.
        validation_folder : dossier de validation.
        output_folder : dossier de sortie.
    Options :
        -h ou --help : affiche l'aide
        -a ou --all : active toutes les métriques.
        -v ou --visualize <label_file> : compute et affiche les sorties du réseau, en utilisant la correspondance rgb/étiquettes donnée dans le fichier spécifié
"""
from __future__ import absolute_import, print_function, division

import getopt
from os.path import join, isfile
from os import mkdir
import sys

def main():
    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'hav:',['help','all','visualize='])
    except getopt.error as msg:
        print(msg)
        print(__doc__)
        sys.exit(-1)
#         TODO : finir d'écrire ce gentil petit script...

if __name__ == '__main__':
    main()