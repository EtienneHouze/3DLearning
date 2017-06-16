"""
Script fait pour apprendre depuis les images générées par un modèle 3D.
Syntaxe :
    <modele> <folder_path> <training_folder> <num_classes> [<validation folder>]
        model : name of the net to build, see src.models
        folder_path : directory where the model will be saved.
        training folder : folder containg training data
        num_classes : number of classes
        [optional] valideation_folder : Dossier contenant les données de validation

Options :
    -h ou --help : affiche l'aide
    -l ou --load <file> : charge les poids donnés dans le fichier spécifié.
    -f ou --freeze <nom> : gèle les layers contenant le nom spécifié pour la phase d'apprentissage.
    -e ou --epochs <nombre> : spécifie le nombre d'epochs, 30 par défaut
    -b ou --batch <nombre> : spécifie la taille d'un batch, 5 par défaut
    -d ou --decay <valeur> : spécifie un taux de diminution du learning rate, aucun par défault.
    -r ou --rate <taux> : définit le taux d'apprentissage spécifié. Par défaut à 1e-4
    -h ou --height <hauteur> : définit la hauteur des images.
    -w ou --width <largeur> : définit la largeur des images.
"""

from __future__ import absolute_import, division, print_function


import getopt
import sys
from os import listdir
from os.path import join, isfile, isdir
from os import mkdir

from src.CityScapeModel import CityScapeModel

def main():
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:],'he:b:f:l:r:',['help','epochs=','batch=','freeze=','load=','rate='])
    except getopt.error as msg:
        print(msg)
        print("Try -h or --help  for help")
        sys.exit(1)

    epochs = 30
    batch_size = 5
    weights = None
    lr_decay = 1
    eta = 0.001
    freeze_name = []
    im_h = 256
    im_w = 256

    for o, a in opts:
        if o in ['-h','--help']:
            print(__doc__)
            sys.exit(0)
        if o == '-e' or o == '--epochs':
            epochs = int(a)
        if o == '-b' or o == '--batch':
            batch_size = int(a)
        if o in ['-l', '--load']:
            weights = a
        if o in ['-d', '--decay']:
            lr_decay = float(a)
        if o in ['-r','--rate']:
            eta = float(a)
        if o in ['-f','--freeze']:
            freeze_name.append(a)
        if o in ['-h','--height']:
            im_h = int(a)
        if o in ['-w','--width']:
            im_w = int(a)

    model_name = args[0]
    root_folder = args[1]
    train_folder = args[2]
    num_classes = int(args[3])
    val_folder = train_folder
    if len(args) == 5:
        val_folder = args[4]

    model = CityScapeModel(root_folder)

    model.define_input((im_h,im_w,5))
    model.define_numlabs(num_classes)
    model.define_network(model_name)

    train_size = 0
    val_size = 0

    # <editor-fold desc="Defining the training and validation sets">
    names = listdir(join(train_folder,'RGB'))
    for name in names:
        if isfile(join(train_folder,'RGB', name)) and 'png' in name:
            train_size += 1
    if val_folder!=train_folder:
        names = listdir(join(val_folder,'RGB'))
        for name in names:
            if isfile(join(val_folder,'RGB,', name)) and 'png' in name:
                val_size += 1
    else:
        val_size = 100
    model.define_training_set(trainset=train_folder,
                              trainsetbuilder='',
                              trainsize=train_size)
    model.define_validation_set(valset=val_folder,
                                valsetbuilder='',
                                valsize=val_size)
    # </editor-fold>

    model.define_loss('categorical_crossentropy')
    model.define_learning_rate(eta)
    model.define_metrics('iou', 'acc')


    model.add_callback('view_output_3D',
                       batch_interval=10,
                       on_epoch=False,
                       num_ins=5)

    model.add_callback(
            'history_loss',
            write_on_epoch=True
    )
    model.add_callback('ckpt')
    model.add_callback('console_display')
    model.add_callback('tensorboard')
    if lr_decay != 1:
        model.add_callback('lr_decay', rate=lr_decay, interval=10)

    model.build_net()
    if weights:
        model.load_weights(weights)
    if len(freeze_name) > 0:
        for layer_name in freeze_name:
            model.freeze_layers_with_name(layer_name)
    model.print_net()
    model.print_png()
    model.save_tojson()

    model.train_3D(epochs=epochs,
                   batch_size=batch_size,
                   save=True
                   )

if __name__=="__main__":
    main()