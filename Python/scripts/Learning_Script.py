# Learning Script

from __future__ import absolute_import, division, print_function

from os.path import join
from src.CityScapeModel import CityScapeModel

DATAFOLDER = 'C:/Users/Etienne.Houze/Documents/EtienneData'
model_name = 'up_skips_meta_pool_drop'

model = CityScapeModel(join(DATAFOLDER,'models',model_name))


# Callbacks ==================================================
model.add_callback('view_output',
                  batch_interval=100,
                  on_epoch=False,
                  num_ins=5)

model.add_callback(
        'history_loss',
        write_on_epoch=True
        )
model.add_callback('tensorboard')
model.add_callback('ckpt')
model.add_callback('console_display')
model.add_callback('lr_decay',rate=0.9, interval=20)
# =======================================================
model.define_input((256, 512, 5))
model.define_numlabs(18)
model.define_network('up_skips_meta_pool_drop')
model.define_training_set(join(DATAFOLDER,'trainmeddisp'), 'with_z', 2975)
model.define_validation_set(join(DATAFOLDER,'val_with_disp'), 'with_z', 10)
model.define_loss('categorical_crossentropy')
model.define_learning_rate(0.001)
model.build_net()
model.define_metrics('iou','acc',)
# # test.print_model()
# model.load_weights(join(DATAFOLDER,'models','up_skips_rgb_only','saves','default'))
# model.freeze_layers_with_name('net')
model.save_tojson()
model.print_png()
model.print_net()
model.train(epochs=100,
           batch_size=5,
           save=True
           )
# test.compute_on_dir('D:/EtienneData/train_with_disp', outdir=join(DATAFOLDER,'output'))
# # test.evaluate(outputfile='./results_val.csv')