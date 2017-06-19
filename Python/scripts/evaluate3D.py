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

import csv
import getopt
from os.path import join, isfile, isdir
from os import mkdir

import time

from helpers.preprocess import dir_size
from helpers.BatchGenerator import BatchGenerator
from src.CityScapeModel import CityScapeModel
import sys

def main():
    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'hav:',['help','all','visualize='])
    except getopt.error as msg:
        print(msg)
        print(__doc__)
        sys.exit(-1)


    all_metrics = False

    # TODO : trouver comment mettre en place la visualisation
    visu_folder = None

    for o,a in opts:
        if o in ['-h','--help']:
            print(__doc__)
            sys.exit(1)
        if o in ['-a','--all']:
            all_metrics = True
        if o in ['-v','--visualize']:
            visu_folder = a

    model_folder = args[0]
    validation_folder = args[1]
    output_folder = args[2]
    model = CityScapeModel(model_folder)

    if not isdir(output_folder):
        mkdir(output_folder)

    if all_metrics:
        mets = ['acc','iou']
        for i in range(model.prop_dict['num_labs']):
            mets.append('cat-iou_'+str(i))
        model.define_metrics(*mets)

    testdirsize = dir_size(join(validation_folder,'RGB'))
    test_gen = BatchGenerator(traindir=validation_folder,
                              traindirsize=testdirsize,
                              trainsetsize=testdirsize,
                              batchsize=1)
    gen = test_gen.generate_batch_for_3D()
    counter = 0

    model.compile()
    model.load_weights()
    out_dict_list = []
    for (x_test, y_test) in gen:
        if counter < testdirsize:
            line = {}
            begin_time = time.time()
            values = model.model.test_on_batch(x_test,y_test)
            end_time = time.time()
            line['loss'] = values[0]
            line['time'] = end_time - begin_time
            for i in range(1,len(values)):
                line[model.prop_dict['metrics'][i-1]] = values[i]
            print(line)
            out_dict_list.append(line)
            counter += 1
        else:
            break
    with open(join(output_folder,'raw_output.csv'),'w') as f:
        writer = csv.DictWriter(f, out_dict_list[0].keys())
        writer.writeheader()
        writer.writerows(out_dict_list)
    means = out_dict_list[0]
    for key in means.keys():
        for i in range(1,testdirsize):
            means[key] += out_dict_list[i][key]
        means[key] /= testdirsize
    with open(join(output_folder,'mean_output.csv'),'w') as f:
        writer = csv.DictWriter(f, means.keys())
        writer.writeheader()
        writer.writerow(means)

    # if visu_folder:




if __name__ == '__main__':
    main()