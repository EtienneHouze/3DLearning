"""This script launches an automatic procedure for training a neural network.
The syntax of the command is :
    <network> <folder_path> <training_folder> <validation_folder> <training_type> <height> <width> <depth> <num_classes>
        network : name of the net building function. See src.models.
        folder_path : specifies the folder in which to store the model
        training_folder : path to the training folder
        validation_folder : path to the validation folder
        training_type : one of the available options for input data, see batchgenerator
        height, width, depth : dimensionrs of the inputs
        num_classes : number of classes

    options :
        -h or --help for help
        -e or --epochs <number of epcohs> : specifies the nuumber of epochs, default is 30.
        -b or --batch <batch_size> : defines the size of a batch
        -l or --load <path> : load weigths from the specified path
        -d or --decay <rate> : adds learning rate decay with specified rate
        -r or --rate <learning> : applies the specified leanring rate, default value is 1e-4
        -f or --freeze <layer_name> : freeze the specified layer in the training process.

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
        opts, args = getopt.gnu_getopt(sys.argv[1:],'he:b:d:f:l:f:',['help','epochs=','batch=','decay=','load=','freeze='])
    except getopt.error as msg:
        print(msg)
        print ("try -h or --help for help")
        sys.exit(1)

    epochs = 30
    batch_size = 5
    weights = None
    lr_decay = 1
    eta = 0.001
    freeze_name = []

    for o, a in opts:
        if o == '-e' or o == '--epochs':
            epochs = int(a)
        if o == '-b' or o == '--batch':
            batch_size = int(a)
        if o in ['-l', '--load']:
            weights = a
        if o in ['-d', '--decay']:
            lr_decay = float(a)
        if o in ['-h', '--help']:
            print(__doc__)
            sys.exit(2)
        if o in ['-r','--rate']:
            eta = float(a)
        if o in ['-f','--freeze']:
            freeze_name.append(a)

    network, folder_path, training_folder, validation_folder, training_type, height, width, depth, num_classes = None, None, None, None, None, None, None, None, None
    if len(args) == 9:
        network = args[0]
        folder_path = args[1]
        training_folder = args[2]
        validation_folder = args[3]
        training_type = args[4]
        height = int(args[5])
        width = int(args[6])
        depth = int(args[7])
        num_classes = int(args[8])
    else:
        print("invalid number of arguments")
        print("try -h or --help for help")
        sys.exit(1)




    model = CityScapeModel(folder_path)



    model.define_input((height,width,depth))
    model.define_numlabs(num_classes)
    model.define_network(network)
    # <editor-fold desc="Defining the training and validation sets">
    names = listdir(training_folder)
    train_size = 0
    val_size = 0
    for name in names:
        if isfile(join(training_folder,name)) and 'im'in name:
            train_size += 1
    names = listdir(validation_folder)
    for name in names:
        if isfile(join(validation_folder,name)) and 'im'in name:
            val_size += 1
    model.define_training_set(trainset=training_folder,
                              trainsetbuilder=training_type,
                              trainsize=train_size)
    model.define_validation_set(valset=validation_folder,
                                valsetbuilder=training_type,
                                valsize=val_size)
    # </editor-fold>
    model.define_loss('categorical_crossentropy')
    model.define_learning_rate(eta)
    model.define_metrics('iou','acc')
    # <editor-fold desc="Callback functions">
    model.add_callback('view_output',
                       batch_interval=10,
                       on_epoch=False,
                       num_ins=5)

    model.add_callback(
            'history_loss',
            write_on_epoch=True
    )
    model.add_callback('ckpt')
    model.add_callback('console_display')
    if lr_decay != 1:
        model.add_callback('lr_decay',rate=lr_decay,interval=10)
    # </editor-fold>
    model.build_net()
    if weights:
        model.load_weights(weights)
    if len(freeze_name) > 0:
        for layer_name in freeze_name:
            model.freeze_layers_with_name(layer_name)
    model.print_net()
    model.print_png()
    model.save_tojson()

    model.train(epochs=epochs,
                batch_size=batch_size,
                save=True
                )




if __name__ == "__main__":
    main()