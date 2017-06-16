from __future__ import absolute_import, division, print_function

from os.path import join
from os import listdir

import numpy as np
from PIL import Image
from helpers.preprocess import dir_size
"""
    A class to define a batch generator used in training to avoid manipulationg huge np arrays.
"""

class BatchGenerator:
    def __init__(self, traindir, city_model, trainsetsize, batchsize=5, traindirsize=2975):
        """
        Initialize a BatchGenerator object
        :param 
            traindir: path to the training folder 
            city_model: reference to the CityScapeModel
            trainsetsize: size of the set size to use for training
            batchsize: size of the batch to generate
            traindirsize: number of images in the training dir. Leave to default in most cases...
        """
        self.i = 0
        self.indices = np.random.choice(a=traindirsize,
                                        size=trainsetsize,
                                        replace=False
                                        )
        self.traindir = traindir
        self.batchsize = batchsize
        self.city_model = city_model
        self.epoch_size = trainsetsize // self.batchsize

    def generate_batch(self, option = ''):
        """
        Generator of a batch.
        :param 
            option: a string defining the option to apply to the batch. It can be :
                * '' or 'without_disp' : default case, yields inputs with 3 channels
                * 'with_disp' : yields inputs with 4 channels, containing data from disparity.
                * 'with_z' : yields a 5-channel input to store the height of the pixel
                * 'resize4' : yields a 4 times reduced label image
        :return
            a tuple (ins, labs) of np arrays corresponding to a batch.
        """
        if option=='with_disp':
            while self.i > -1:
                i = self.i % self.epoch_size
                ins_list = []
                labs_list = []
                if (i == 0):
                    np.random.shuffle(self.indices)
                for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                    Im = Image.open(join(self.traindir, '_' + str(k) + '_im_.png'))
                    Label = Image.open(join(self.traindir, '_' + str(k) + '_lab_.png'))
                    Disp = Image.open(join(self.traindir, '_' + str(k) + '_disp_.png'))
                    im = np.asarray(Im, dtype=np.float32)
                    lab = np.asarray(Label.convert(mode="L"), dtype=np.int)
                    disp = np.asarray(Disp, dtype=np.float32)
                    disp = np.expand_dims(disp, axis=2)
                    maxlabs = self.city_model.prop_dict['num_labs'] * np.ones_like(lab)
                    lab = np.minimum(lab, maxlabs)
                    lab = np.eye(self.city_model.prop_dict['num_labs'] + 1)[lab]
                    ins_list.append(np.concatenate((im, disp), axis=-1))
                    labs_list.append(lab)
                self.i += 1
                yield (np.asarray(ins_list), np.asarray(labs_list))
        elif option == 'with_z':
            while self.i > -1:
                i = self.i % self.epoch_size
                ins_list = []
                labs_list = []
                if (i == 0):
                    np.random.shuffle(self.indices)
                height_array = np.arange(start=0,
                                         stop=self.city_model.prop_dict['input_shape'][0])
                height_array = np.expand_dims(height_array,
                                              axis=1)
                height_array = np.tile(height_array,
                                       reps=(1,self.city_model.prop_dict['input_shape'][1]))
                height_array = np.expand_dims(height_array,
                                              axis=2)
                for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                    Im = Image.open(join(self.traindir, '_' + str(k) + '_im_.png'))
                    Label = Image.open(join(self.traindir, '_' + str(k) + '_lab_.png'))
                    Disp = Image.open(join(self.traindir, '_' + str(k) + '_disp_.png'))
                    im = np.asarray(Im, dtype=np.float32)
                    lab = np.asarray(Label.convert(mode="L"), dtype=np.int)
                    disp = np.asarray(Disp, dtype=np.float32)
                    disp = np.expand_dims(disp, axis=2)
                    maxlabs = self.city_model.prop_dict['num_labs'] * np.ones_like(lab)
                    lab = np.minimum(lab, maxlabs)
                    lab = np.eye(self.city_model.prop_dict['num_labs'] + 1)[lab]
                    ins_list.append(np.concatenate((im, disp,height_array), axis=-1))
                    labs_list.append(lab)
                self.i += 1
                yield (np.asarray(ins_list), np.asarray(labs_list))
        elif option == 'resize4':
            while self.i > -1:
                i = self.i % self.epoch_size
                ins_list = []
                labs_list = []
                if (i == 0):
                    np.random.shuffle(self.indices)
                for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                    Label = Image.open(join(self.traindir, '_' + str(k) + '_lab_.png'))
                    lab = np.asarray(Label.convert(mode="L"), dtype=np.int)
                    Label.thumbnail((128, 64))
                    im = np.asarray(Label.convert(mode='L'), dtype=np.int)
                    maxlabs = self.city_model.prop_dict['num_labs'] * np.ones_like(lab)
                    mini_maxlabs = self.city_model.prop_dict['num_labs'] * np.ones_like(im)
                    lab = np.minimum(lab, maxlabs)
                    lab = np.eye(self.city_model.prop_dict['num_labs'] + 1)[lab]
                    im = np.minimum(im, mini_maxlabs)
                    im = np.eye(self.city_model.prop_dict['num_labs'] + 1)[im]
                    ins_list.append(im)
                    labs_list.append(lab)
                self.i += 1
                yield (np.asarray(ins_list), np.asarray(labs_list))
        elif option == 'without_disp' or option == '':
            while self.i > -1:
                i = self.i % self.epoch_size
                ins_list = []
                labs_list = []
                if (i == 0):
                    np.random.shuffle(self.indices)
                for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                    Im = Image.open(join(self.traindir, '_' + str(k) + '_im_.png'))
                    Label = Image.open(join(self.traindir, '_' + str(k) + '_lab_.png'))
                    im = np.asarray(Im, dtype=np.float32)
                    lab = np.asarray(Label.convert(mode="L"), dtype=np.int)
                    maxlabs = self.city_model.prop_dict['num_labs'] * np.ones_like(lab)
                    lab = np.minimum(lab, maxlabs)
                    lab = np.eye(self.city_model.prop_dict['num_labs'] + 1)[lab]
                    ins_list.append(im)
                    labs_list.append(lab)
                self.i += 1
                yield (np.asarray(ins_list), np.asarray(labs_list))
        else:
            print("Invalid option !")

    def generate_input_only(self, option = ''):
        if option == 'with_disp':
            while self.i > -1:
                i = self.i % self.epoch_size
                ins_list = []
                if (i == 0):
                    np.random.shuffle(self.indices)
                for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                    Im = Image.open(join(self.traindir, '_' + str(k) + '_im_.png'))
                    Disp = Image.open(join(self.traindir, '_' + str(k) + '_disp_.png'))
                    im = np.asarray(Im, dtype=np.float32)
                    disp = np.asarray(Disp, dtype=np.float32)
                    disp = np.expand_dims(disp, axis=2)
                    lab = np.minimum(lab, maxlabs)
                    ins_list.append(np.concatenate((im, disp), axis=-1))
                self.i += 1
                yield np.asarray(ins_list)
        elif option == 'with_z':
            while self.i > -1:
                i = self.i % self.epoch_size
                ins_list = []
                if (i == 0):
                    np.random.shuffle(self.indices)
                height_array = np.arange(start=0,
                                         stop=self.city_model.prop_dict['input_shape'][0])
                height_array = np.expand_dims(height_array,
                                              axis=1)
                height_array = np.tile(height_array,
                                       reps=(1, self.city_model.prop_dict['input_shape'][1]))
                height_array = np.expand_dims(height_array,
                                              axis=2)
                for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                    Im = Image.open(join(self.traindir, '_' + str(k) + '_im_.png'))
                    Disp = Image.open(join(self.traindir, '_' + str(k) + '_disp_.png'))
                    im = np.asarray(Im, dtype=np.float32)
                    disp = np.asarray(Disp, dtype=np.float32)
                    disp = np.expand_dims(disp, axis=2)
                    ins_list.append(np.concatenate((im, disp, height_array), axis=-1))
                self.i += 1
                yield np.asarray(ins_list)
        elif option == 'without_disp' or option == '':
            while self.i > -1:
                i = self.i % self.epoch_size
                ins_list = []
                if (i == 0):
                    np.random.shuffle(self.indices)
                for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                    Im = Image.open(join(self.traindir, '_' + str(k) + '_im_.png'))
                    im = np.asarray(Im, dtype=np.float32)
                    ins_list.append(im)
                self.i += 1
                yield np.asarray(ins_list)
        else:
            print("Invalid option !")

    def generate_batch_for_3D(self):

        self.lab_dir = join(self.traindir,'Labels')
        self.rgb_dir = join(self.traindir,'RGB')
        self.alt_dir = join(self.traindir,'Altitude')
        self.depth_dir = join(self.traindir,'Depth')

        self.name_list = listdir(self.rgb_dir)

        while self.i > -1:
            i = self.i % self.epoch_size
            ins_list = []
            labs_list = []
            if (i == 0):
                np.random.shuffle(self.indices)
            for k in self.indices[self.batchsize * i: (i * self.batchsize) + self.batchsize]:
                rgb = Image.open(join(self.rgb_dir,self.name_list[k]))
                lab = Image.open(join(self.lab_dir,self.name_list[k]))
                depth = Image.open(join(self.depth_dir, self.name_list[k]))
                alt = Image.open(join(self.alt_dir,self.name_list[k]))
                rgb_array = np.asarray(rgb)
                lab_array = np.asarray(lab)
                # maxlabs = self.city_model.prop_dict['num_labs'] * np.ones_like(lab_array)
                lab_array = np.eye(self.city_model.prop_dict['num_labs']+1)[lab_array]
                labs_list.append(lab_array)
                depth_array = np.asarray(depth)
                depth_array = np.expand_dims(depth_array,axis=2)
                alt_array = np.asarray(alt)
                alt_array = np.expand_dims(alt_array,axis=2)
                ins_list.append(np.concatenate((rgb_array,depth_array,alt_array),axis=-1))
            self.i += 1
            yield (np.asarray(ins_list), np.asarray(labs_list))