import tensorflow as tf
from Model import *
from utils import *
import numpy as np


test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_acc = tf.keras.metrics.CategoricalAccuracy('test_acc', dtype=tf.float32)
domain_source_loss = tf.keras.metrics.Mean('domain_source_loss', dtype=tf.float32)
domain_target_loss = tf.keras.metrics.Mean('domain_target_loss', dtype=tf.float32)
gen_loss = tf.keras.metrics.Mean('generator_source_loss', dtype=tf.float32)
disc_loss = tf.keras.metrics.Mean('discriminator_t_loss', dtype=tf.float32)
source_test_accuracy = tf.keras.metrics.CategoricalAccuracy('source_test_accuracy')
target_test_accuracy = tf.keras.metrics.CategoricalAccuracy('target_test_accuracy')


def test_step1(encoder_s, encoder_t, classifier, source_batch, target_batch):
    Xs, Ys = get_data_from_batch(source_batch)
    Xt, Yt = get_data_from_batch(target_batch)
    feature_s = encoder_s(Xs, training=False)
    feature_t = encoder_t(Xt, training=False)
    prediction_s = classifier(feature_s, training=False)
    prediction_t = classifier(feature_t, training=False)
    loss = (classifier_loss(prediction_s, Ys).numpy() + classifier_loss(prediction_t, Yt).numpy()) / 2
    test_loss(loss)
    source_test_accuracy(Ys, prediction_s)
    target_test_accuracy(Yt, prediction_t)


def test_step2(encoder_s, encoder_t, discriminator_domain,
               source_batch, target_batch):
    Xs, Ys = get_data_from_batch(source_batch)
    Xt, Yt = get_data_from_batch(target_batch)
    feature_s = encoder_s(Xs, training=False)
    feature_t = encoder_t(Xt, training=False)
    source_decision = discriminator_domain(feature_s, training=False)
    target_decision = discriminator_domain(feature_t, training=False)
    domain_source_loss(discriminator_loss(source_decision, target_decision).numpy())
    domain_target_loss(discriminator_loss(target_decision, source_decision).numpy())


def test_step3(generator_s, generator_t, discriminator_t, discriminator_s,
               source_batch, target_batch):
    Xs, Ys = get_data_from_batch(source_batch)
    Xt, Yt = get_data_from_batch(target_batch)
    generated_t = generator_s(Xs, training=False)
    real_target_output = discriminator_t(Xt, training=False)
    fake_target_output = discriminator_t(generated_t, training=False)
    gen_s_loss = generator_loss(fake_target_output)
    disc_t_loss = discriminator_loss(real_target_output, fake_target_output)

    gen_loss(gen_s_loss)
    disc_loss(disc_t_loss)


def test_step4(generator_s, encoder_t, classifier, source_batch):
    Xs, Ys = get_data_from_batch(source_batch)
    generated_t = generator_s(Xs, training=False)
    feature_t = encoder_t(generated_t, training=False)
    prediction = classifier(feature_t, training=False)
    loss = classifier_loss(prediction, Ys)
    test_loss(loss)
    test_acc(Ys, prediction)
