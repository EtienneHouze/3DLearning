import csv
from os.path import join

import time
import numpy as np
from PIL import Image
from helpers.BatchGenerator import BatchGenerator
from keras.callbacks import Callback
import keras.backend as K


# TODO : Create callback functions to do stuff...

class ViewOutput(Callback):
    """
    Allows user to save an output of the network every specified step as a png file.
    """

    def __init__(self, citymodel, options):
        """
        :param 
            citymodel: the cityscape model 
            options: dictionnary of the options. Can contain :
                * 'batch_interval' : an int, how ofter we want to save an image. Default is 0.
                * 'num_ins' : number of inputs to compute. For now, only 1 is relevant.
                * 'on_epoch' : a bool, whether to print on every epoch or not. Default is True.
        """
        super(ViewOutput, self).__init__()
        self.citymodel = citymodel
        self.batch_interval = 0
        self.i = 0
        self.epoch_counter = 0
        self.num_ins = 1
        self.on_epoch = True
        self.gen = None
        self.x = None
        if ('batch_interval' in options.keys()):
            self.batch_interval = options['batch_interval']
        if ('num_ins' in options.keys()):
            self.num_ins = options['num_ins']
        if ('on_epoch' in options.keys()):
            self.on_epoch = options['on_epoch']

    def on_train_begin(self, logs={}):
        if len(self.citymodel.prop_dict['valset'])>0:
            self.gen = BatchGenerator(traindir=self.citymodel.prop_dict['valset'][1],
                                      city_model=self.citymodel,
                                      trainsetsize=self.citymodel.prop_dict['valset'][2],
                                      batchsize=self.num_ins,
                                      traindirsize=self.citymodel.prop_dict['valset'][2])
            self.x, _ = next(self.gen.generate_batch(option=self.citymodel.prop_dict['valset'][0]))
        else:
            self.gen = BatchGenerator(traindir=self.citymodel.prop_dict['trainset'][1],
                                      city_model=self.citymodel,
                                      trainsetsize=self.citymodel.prop_dict['trainset'][2],
                                      batchsize=self.num_ins)
            self.x, _ = next(self.gen.generate_batch(option=self.citymodel.prop_dict['trainset'][0]))
        for i in range(self.num_ins):
            bob = self.x[i, :, :, :]
            bob = bob[:, :, 0:3].astype(np.uint8)
            Input = Image.fromarray(bob)
            Input.save(join(self.citymodel.prop_dict['directory'], 'watch', self.citymodel.prop_dict['name'] + str(i) + '_input.png'))

    def on_batch_end(self, batch, logs={}):
        if self.batch_interval != 0:
            self.i = self.i + 1
            if self.i % self.batch_interval == 0:
                y = self.citymodel.model.predict_on_batch(self.x)
                y = np.argmax(y,axis=3)
                for i in range(self.num_ins):
                    Output = Image.fromarray(y[i, :, :].astype(np.uint8))
                    Output.save(join(self.citymodel.prop_dict['directory'], 'watch',
                                     self.citymodel.prop_dict['name'] + str(i) + '_output_' + str(self.epoch_counter) + '_' + str(self.i) + '_.png'))

    def on_epoch_end(self, epoch, logs={}):
        if (self.on_epoch):
            y = self.citymodel.model.predict_on_batch(self.x)
            y = np.argmax(y, axis=3)
            for i in range(self.num_ins):
                Output = Image.fromarray(y[i, :, :].astype(np.uint8))
                Output.save(join(self.citymodel.prop_dict['directory'], 'watch',
                                 self.citymodel.prop_dict['name'] + str(i)+ '_output_epoch_' + str(self.epoch_counter) + '_.png'))
        self.epoch_counter += 1

class ViewOutput_3D(Callback):
    """
    Allows user to save an output of the network every specified step as a png file.
    """

    def __init__(self, citymodel, options):
        """
        :param
            citymodel: the cityscape model
            options: dictionnary of the options. Can contain :
                * 'batch_interval' : an int, how ofter we want to save an image. Default is 0.
                * 'num_ins' : number of inputs to compute. For now, only 1 is relevant.
                * 'on_epoch' : a bool, whether to print on every epoch or not. Default is True.
        """
        super(ViewOutput_3D, self).__init__()
        self.citymodel = citymodel
        self.batch_interval = 0
        self.i = 0
        self.epoch_counter = 0
        self.num_ins = 1
        self.on_epoch = True
        self.gen = None
        self.x = None
        if ('batch_interval' in options.keys()):
            self.batch_interval = options['batch_interval']
        if ('num_ins' in options.keys()):
            self.num_ins = options['num_ins']
        if ('on_epoch' in options.keys()):
            self.on_epoch = options['on_epoch']

    def on_train_begin(self, logs={}):
        if len(self.citymodel.prop_dict['valset'])>0:
            self.gen = BatchGenerator(traindir=self.citymodel.prop_dict['valset'][1],
                                      city_model=self.citymodel,
                                      trainsetsize=self.citymodel.prop_dict['valset'][2],
                                      batchsize=self.num_ins,
                                      traindirsize=self.citymodel.prop_dict['valset'][2])
            self.x, _ = next(self.gen.generate_batch_for_3D())
        else:
            self.gen = BatchGenerator(traindir=self.citymodel.prop_dict['trainset'][1],
                                      city_model=self.citymodel,
                                      trainsetsize=self.citymodel.prop_dict['trainset'][2],
                                      batchsize=self.num_ins)
            self.x, _ = next(self.gen.generate_batch_for_3D())
        for i in range(self.num_ins):
            bob = self.x[i, :, :, :]
            bob = bob[:, :, 0:3].astype(np.uint8)
            Input = Image.fromarray(bob)
            Input.save(join(self.citymodel.prop_dict['directory'], 'watch', self.citymodel.prop_dict['name'] + str(i) + '_input.png'))

    def on_batch_end(self, batch, logs={}):
        if self.batch_interval != 0:
            self.i = self.i + 1
            if self.i % self.batch_interval == 0:
                y = self.citymodel.model.predict_on_batch(self.x)
                y = np.argmax(y,axis=3)
                for i in range(self.num_ins):
                    Output = Image.fromarray(y[i, :, :].astype(np.uint8))
                    Output.save(join(self.citymodel.prop_dict['directory'], 'watch',
                                     self.citymodel.prop_dict['name'] + str(i) + '_output_' + str(self.epoch_counter) + '_' + str(self.i) + '_.png'))

    def on_epoch_end(self, epoch, logs={}):
        if (self.on_epoch):
            y = self.citymodel.model.predict_on_batch(self.x)
            y = np.argmax(y, axis=3)
            for i in range(self.num_ins):
                Output = Image.fromarray(y[i, :, :].astype(np.uint8))
                Output.save(join(self.citymodel.prop_dict['directory'], 'watch',
                                 self.citymodel.prop_dict['name'] + str(i)+ '_output_epoch_' + str(self.epoch_counter) + '_.png'))
        self.epoch_counter += 1



class LossHistory(Callback):
    """
    Allows to record history of the loss with set frequency.
    """

    def __init__(self, citymodel, options):
        super(LossHistory, self).__init__()
        self.citymodel = citymodel
        self.losses = None
        self.i = 0
        self.timer = 0
        self.begin_time = 0
        self.frequency = 1
        self.write = True
        if 'frequency' in options.keys():
            self.frequency = options['frequency']
        if 'write_on_epoch' in options.keys():
            self.write = options['write_on_epoch']

    def on_batch_begin(self, batch, logs=None):
        self.begin_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.i += 1
        if self.i % self.frequency == 0:
            line = {}
            line = logs
            line['time'] = time.time()-self.begin_time
            self.losses.append(line)

    def on_epoch_end(self, epoch, logs=None):
        if self.write:
            with open(join(self.citymodel.prop_dict['directory'], 'logs',
                           self.citymodel.prop_dict['name'] + '_losses.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, self.losses[0].keys())
                if epoch==0:
                    writer.writeheader()
                writer.writerows(self.losses)


class ConsoleDisplay(Callback):
    """
    A callback function to be very explicit in the console log.
    """
    def __init__(self,citymodel,options):
        """
        :param citymodel: 
        :type citymodel: 
        :param options: optional arguments:
            
        :type options: a dictionary
        """
        super(ConsoleDisplay,self).__init__()
        self.citymodel = citymodel
        self.iteration = 0
        self.epoch_count = 1

    def on_batch_end(self, batch, logs=None):
        print("Batch " + str(batch) + " of epoch " + str(self.epoch_count))
        print("loss : " + str(logs['loss']))
        print("============")
        self.iteration += 1

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1


class LearningRateDecay(Callback):
    """
    Creates a callback to decrease learning rate during learning process
    """
    def __init__(self, citymodel, options):
        """
        :param citymodel: reference to the model 
        :type citymodel: a CityScapeModel
        :param options: different optionnal args
            * 'rate': coefficient by which the learning rate is multiplied.
            *  'interval' : interval, in epochs, of the decreasing.
        :type options: dictionnary
        """
        super(LearningRateDecay,self).__init__()
        self.citymodel = citymodel
        self.rate = 1
        self.interval = 1
        if 'rate' in options.keys():
            self.rate = options['rate']
        if 'interval' in options.keys():
            self.interval = options['interval']

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.interval == 0:
            old_lr = K.get_value(self.citymodel.model.optimizer.lr)
            new_lr = self.rate*old_lr
            K.set_value(self.citymodel.model.optimizer.lr, new_lr)

# A dictionnary linking functions to their names.
callbacks_dict = {
    'view_output': ViewOutput,
    'view_output_3D': ViewOutput_3D,
    'history_loss': LossHistory,
    'console_display': ConsoleDisplay,
    'lr_decay': LearningRateDecay
}
