""" This script directly evaluates a model imported from a folder over a specified set of images.
    The syntax is : <model_folder> <test_folder> <output_folder>
        model_folder : path to the folder containing the model
        test_folder : path to the folder containing images and labels for testing
        output_folder : path to the folder where results will be stored.

    options :
        -h or --help : displays help
        -a or --all : adds iou_metrics for all IoU

"""

from __future__ import absolute_import, division, print_function

import getopt
import sys
from os import listdir
from os.path import join, isfile, isdir
from os import mkdir
import csv
import time

from src.CityScapeModel import CityScapeModel
from helpers.BatchGenerator import BatchGenerator

def main():
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:],'ha',['help','all'])
    except getopt.error as msg:
        print(msg)
        print("for help, try -h or --help")
        sys.exit(1)

    all_metrics = False

    for o,a in opts:
        if o in ['-h','--help']:
            print(__doc__)
            sys.exit(2)
        if o in ['-a','--all']:
            all_metrics = True

    if len(args) != 3:
        print("Incorrect number of arguments, see -h or --help")
        sys.exit(1)

    model_folder = args[0]
    test_folder = args[1]
    output_folder = args[2]
    model = CityScapeModel(model_folder)

    if not isdir(output_folder):
        mkdir(output_folder)

    if all_metrics :
        mets = ['acc','iou']
        for i in range(model.prop_dict['num_labs']):
            mets.append('cat-iou_' + str(i))
        model.define_metrics(*mets)
    testdirsize = len([name for name in listdir(test_folder) if isfile(join(test_folder,name)) and 'im' in name])
    test_gen = BatchGenerator(traindir=test_folder,
                              city_model=model,
                              traindirsize=testdirsize,
                              trainsetsize=testdirsize,
                              batchsize=1)
    gen = test_gen.generate_batch(option=model.prop_dict['valset'][0])
    counter = 0

    model.compile()
    model.load_weights()
    out_dict_list = []
    for (x_test,y_test) in gen:
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
            counter +=1
        else:
            break
    with open(join(output_folder,'raw_output.csv'),'w') as f:
        writer = csv.DictWriter(f, out_dict_list[0].keys())
        writer.writeheader()
        writer.writerows(out_dict_list)
    means = out_dict_list[0]
    for key in means.keys():
        for i in range(1, testdirsize):
            means[key] += out_dict_list[i][key]
        means[key] /= testdirsize
    with open(join(output_folder,'mean_output.csv'),'w') as f:
        writer = csv.DictWriter(f, means.keys())
        writer.writeheader()
        writer.writerow(means)


if __name__=="__main__":
    main()