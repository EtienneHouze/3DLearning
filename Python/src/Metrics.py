from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf
import keras.backend as K

from helpers.Values import weights18
# from sklearn.metrics import jaccard_similarity_score
# TODO : Trouver un moyen de passer la valeur du nombre de labels en évitant d'appeler le tensuer
def iou(y_true, y_pred):
    """
    
    Args:
        y_true (4D-tensor): ground truth label 
        y_pred (4D-tensor): output of the network (after softmax)

    Returns:
        The value of the mean IOU loss
    """
    # numlabs = y_pred.get_shape()[-1].value
    numlabs = 19
    y_pred = tf.argmax(input=y_pred,
                       axis=-1
                       )
    y_pred = tf.one_hot(indices=y_pred,
                        axis=-1,
                        depth=numlabs)
    equality = tf.cast(tf.equal(y_true, y_pred),
                       dtype=tf.float32)
    intersection = tf.multiply(y_true,equality)
    union = y_pred + y_true - intersection
    nd = intersection.get_shape().ndims
    TP = tf.reduce_sum(intersection,
                       axis=np.arange(start=0, stop=nd - 1, step=1)
                       )
    Neg = tf.maximum(tf.reduce_sum(union,
                                   axis=np.arange(start=0, stop=nd - 1, step=1)
                                   ),
                     1)

    return tf.reduce_mean(TP/Neg,axis=-1)

def weighted_loss_18(output, target):
    weights = tf.constant(weights18, dtype=tf.float32)

    return -tf.reduce_mean(tf.reduce_sum(tf.multiply(target,
                                                     tf.multiply(weights,
                                                                 tf.log(output + K.epsilon()
                                                                        )
                                                                 )
                                                     ),
                                         axis=-1
                                         )
                           )

class Metrics():
    def __init__(self, citymodel, cat = -1):
        self.numlabs = citymodel.prop_dict['num_labs']+1
        self.cat = cat

    def iou(self, y_true, y_pred):
        """

        Args:
            y_true (4D-tensor): ground truth label
            y_pred (4D-tensor): output of the network (after softmax)

        Returns:
            The value of the mean IOU loss
        """
        # numlabs = y_pred.get_shape()[-1].value
        numlabs = self.numlabs
        y_pred = tf.argmax(input=y_pred,
                           axis=-1
                           )
        y_pred = tf.one_hot(indices=y_pred,
                            axis=-1,
                            depth=numlabs)
        equality = tf.cast(tf.equal(y_true, y_pred),
                           dtype=tf.float32)
        intersection = tf.multiply(y_true, equality)
        union = y_pred + y_true - intersection
        nd = intersection.get_shape().ndims
        TP = tf.reduce_sum(intersection,
                           axis=np.arange(start=0, stop=nd - 1, step=1)
                           )
        Neg = tf.maximum(tf.reduce_sum(union,
                                       axis=np.arange(start=0, stop=nd - 1, step=1)
                                       ),
                         1)
        Res = TP/Neg
        if self.cat == -1:
            return tf.reduce_mean(Res, axis=-1)
        else:
            return Res[self.cat]


# List renferencing metrics
valid_metrics = ['iou','weighted_loss'] + ['cat-iou_' + str(i) for i in range(255)]

def create_metrics(metricname, citymodel):
    met = Metrics(citymodel)
    fun = lambda x,y:0
    if metricname == 'iou':
        fun = met.iou
    else:
        split = metricname.split('_')
        if split[0]=='cat-iou':
            met.cat = int(split[1])
            fun = met.iou
        else:
            print("Bad name")
    return fun
