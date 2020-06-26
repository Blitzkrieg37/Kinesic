

import tensorflow as tf
import keras as K
import numpy as np



def mse(y_true,y_pred):
    
    loss = tf.square(y_true-y_pred)
    return tf.reduce_sum(loss,axis=-1)

def bce(y_true,y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    print(tf.shape(y_pred))
    y_pred = tf.maximum(y_pred, 1e-15)
    y_pred = tf.minimum(y_pred, 1-1e-15)
    loss = y_true*tf.math.log(y_pred) + (1-y_true)*tf.math.log(1-y_pred)
    print(loss)
    
    return -tf.reduce_sum(loss,axis=-1)

def Loss(y_true,y_pred):
    '''
    Arguments:
    
    y_true:Numpy array Ground truth prediction of the form (m,h1,w1,classes+5)
    y_pred: Tf tensor of same shape as y_true ;Output of model
    
    '''
    batch_size = tf.cast(tf.shape(y_pred)[0],dtype = tf.float32)
    
    lambda_coord = tf.constant(5.0)
    lambda_noobj = tf.constant(0.5)
    
    obj_mask = tf.cast(tf.equal(y_true[...,4],1),dtype=tf.float32)
    ignore_mask = 1-tf.cast(tf.equal(y_true[...,4],-1),dtype=tf.float32)
    
    loc_loss = lambda_coord*tf.cast(mse(y_true[:,:,:,0:4],y_pred[:,:,:,0:4]),dtype=tf.float32)*obj_mask
    
    obj_loss =  tf.cast(bce(y_true[:,:,:,4:5],y_pred[:,:,:,4:5]),dtype=tf.float32)*obj_mask
    no_obj_loss = tf.cast(bce(y_true[:,:,:,4:5],y_pred[:,:,:,4:5]),dtype=tf.float32)*(1-obj_mask)*ignore_mask*lambda_noobj
    class_loss = tf.cast(bce(y_true[:,:,:,5:],y_pred[:,:,:,5:]),dtype=tf.float32)*obj_mask
    
    
    total_loss = obj_loss + no_obj_loss + class_loss + loc_loss
    
    return tf.reduce_sum(total_loss)    
    





