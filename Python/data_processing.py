# Data processing script

from __future__ import absolute_import, division, print_function

from helpers import visualization
from os.path import join


DATAFOLDER = 'C:/Users/Etienne.Houze/Documents/EtienneData'
model_name = 'up_skips_meta_pool_drop_aggreg'

visualization.visualize_csv(join(DATAFOLDER,'models',model_name,'logs','default_losses.csv'),unique_graph=False,
                            means={'iou':400,'loss':400,'acc':400}
                            )
visualization.convert_labelled_output(join(DATAFOLDER,'models',model_name),num_labs=18)
