import tensorflow as tf
from Model import *
from utils import *
import numpy as np


test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_acc = tf.keras.metrics.CategoricalAccuracy('test_acc', dtype=tf.float32)


def test_step1(classifier_data, target_batch):
    Xt, Yt = get_data_from_batch(target_batch)
    prediction = classifier_data(Xt, training=False)
    loss = classifier_loss(prediction, Yt)
    test_loss(loss)
    test_acc(Yt, prediction)


def test_step4(encoder_t, classifier, target_batch):
    Xt, Yt = get_data_from_batch(target_batch)
    feature_t = encoder_t(Xt, training=False)
    prediction = classifier(feature_t, training=False)
    loss = classifier_loss(prediction, Yt)
    test_loss(loss)
    test_acc(Yt, prediction)
