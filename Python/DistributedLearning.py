# Sandbox script

import numpy as np
from helpers import preprocess
from src.CityScapeModel import CityScapeModel
from os.path import join
from src.Metrics import iou

DATAFOLDER = 'C:/Users/Etienne.Houze/Documents/EtienneData'
test = CityScapeModel(join(DATAFOLDER,'up_without'))
test.add_callback('view_output',
                  batch_interval=50,
                  on_epoch=False,
                  num_ins=5)
test.add_callback('tensorboard')
# test.add_callback('csv')
test.add_callback(
        'history_loss',
        write_on_epoch=True
        )
test.add_callback('console_display')
test.define_input((256, 512, 4))
test.define_numlabs(18)
test.define_network('up_without')
test.define_training_set(join(DATAFOLDER,'trainmeddisp'),'with_disp',2975)
test.define_validation_set(join(DATAFOLDER,'val_with_disp'),'with_disp',100)
test.define_loss('categorical_crossentropy')
test.define_metrics('iou','acc')
test.build_net()
test.print_net()
test.save_tojson()
test.print_model()
# test.load_weights()
test.save_tojson()
test.train(epochs=30,
           batch_size=5,
           save=True
           )
test.evaluate(valdirsize=100,outputfile=join(DATAFOLDER,'results_val.csv'))
# for x in x_train:
#     y = test.compute_output(x)
#     print ("hello world")
# test.train([x_train,y_train],2,5,layer = 1)
# test.define_input((256,512,19))
# test.define_output((256,512,19))
# test.add_network_from_builder('up_mini')
# test.compile(index = None,opt=sgd,loss='categorical_crossentropy')
# test.print_net()
# test.save_net()
# gen = BatchGenerator(x_train,y_train,test,batchsize = 5)
# test.models[0].fit(x_train,y_train,epochs=10,batch_size=5,verbose=2)
# y_pred = test.models[0].predict_on_batch(x_train[0:5,:,:,:])
# test.models[1].fit_generator(gen.generate_batch(1),steps_per_epoch = 20, epochs = 1, verbose = 2)
# test.save_model()
# print('done')

# preprocess.produce_training_dir_with_disp(imdir = 'D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',
#                                          labeldir = 'D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',
#                                          dispdir = 'D:/EtienneData/Cityscapes/disparity_trainvaltest/disparity/train',
#                                          outdir = 'D:/EtienneData/train_with_disp',
#                                          imW = 512,
#                                          imH = 256,
#                                          crop = False,
#                                          training_set_size = 2975,
#                                          alllabels = False
#                                          )
