# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:16:11 2019
@author: viryl
"""
from __future__ import print_function, division
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from param import *
import tensorflow_addons as tfa

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def normalized(data):
    data = data.astype(float)
    data -= np.min(data)
    data /= np.max(data)
    return data


def gen_dataset_from_dict(file_dict, Val=False):
    data = file_dict['data']
    data = np.transpose(data, (0, 2, 1))
    label = file_dict['gt']
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=TEST_FRAC, random_state=42)
    if Val:
        data_test, data_val, label_test, label_val = train_test_split(data_test, label_test, test_size=VAL_FRAC,
                                                                      random_state=43)
        val_size = data_val.shape[0]
    data_train = tf.data.Dataset.from_tensor_slices(data_train)
    data_test = tf.data.Dataset.from_tensor_slices(data_test)
    label_train = tf.data.Dataset.from_tensor_slices(label_train)
    label_test = tf.data.Dataset.from_tensor_slices(label_test)
    if Val:
        data_val = tf.data.Dataset.from_tensor_slices(data_val)
        label_val = tf.data.Dataset.from_tensor_slices(label_val)

        val_ds = tf.data.Dataset.zip((data_val, label_val))
        val_ds = val_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(val_size)

    train_ds = tf.data.Dataset.zip((data_train, label_train))
    test_ds = tf.data.Dataset.zip((data_test, label_test))

    train_ds = train_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    if Val:
        return train_ds, test_ds, val_ds
    else:
        return train_ds, test_ds


def plot_result(generator, source_ds, target_ds, procedure):
    source = source_ds.as_numpy_iterator().next()
    x_s, y_s = source['data'], source['label'][0]
    x_t, y_t = [], []
    target = target_ds.as_numpy_iterator().next()
    for i in range(100):
        y_t = target['label'][i]
        if (y_s == y_t).all():
            x_t = target['data'][i]
            break
    x_fake = generator(x_s, training=False)[0]
    x_s = x_s[0]
    plt.subplot(1, 1, 1)
    plt.plot(np.arange(BANDS), x_s, label='source data')
    plt.plot(np.arange(BANDS), x_t, label='target_data')
    plt.plot(np.arange(BANDS), x_fake, label='generated_data')
    plt.legend(loc='lower right')
    plt.title(f'Generate effect on exp{procedure}')
    plt.savefig('./pics/image_at_exp_{:04d}.png'.format(procedure))
    plt.show()


def generate_and_save_Images(model, epoch, test_input):
    """Notice `training` is set to False.
       This is so all layers run in inference mode (batch norm)."""
    """To-do: reshape the curves as they were normalized"""
    prediction = model(test_input, training=False)
    plt.plot(np.arange(72), prediction[0, :, 0])
    plt.savefig('./pics/image_at_{:04d}_epoch.png'.format(epoch))
    plt.show()


def get_data_from_batch(batches):
    return batches['data'], batches['label']


def calculate_acc(target_test_ds,
                  classifier,
                  epoch):
    target_batch = target_test_ds.shuffle(BUFFER_SIZE).as_numpy_iterator().next()
    target_data, target_label = get_data_from_batch(target_batch)
    prediction_t = classifier(target_data, training=False)
    accuracy_t = tf.metrics.Accuracy()
    accuracy_t.update_state(y_true=target_label,
                            y_pred=prediction_t)
    print('Target accuracy for epoch {} is'.format(epoch + 1),
          '{}%'.format(accuracy_t.result().numpy() * 100))


def plot_acc_loss(acc, gen_loss, disc_loss, cls_loss,
                  generator_loss, discriminator_loss, classifier_loss,
                  source_test_ds, target_test_ds,
                  generator, discriminator, classifier,
                  epoch):
    g_loss, d_loss, c_loss, a = [], [], [], []
    for source_test_batch in source_test_ds.as_numpy_iterator():
        for target_test_batch in target_test_ds.as_numpy_iterator():
            X_s, Y_s = get_data_from_batch(source_test_batch)
            X_t, Y_t = get_data_from_batch(target_test_batch)
            generated_target = generator(X_s, training=False)
            real_decision = discriminator(X_t, training=False)
            fake_decision = discriminator(generated_target, training=False)
            prediction = classifier(X_t, training=False)
            accuracy_t = tf.metrics.Accuracy()
            accuracy_t.update_state(y_true=Y_t,
                                    y_pred=prediction)
            a.append(accuracy_t.result().numpy())
            c_loss.append(classifier_loss(prediction, Y_t).numpy())
            g_loss.append(generator_loss(fake_decision).numpy())
            d_loss.append(discriminator_loss(real_decision, fake_decision).numpy())
    a = np.average(a)
    acc.append(a)
    cls_loss.append(np.average(c_loss))
    gen_loss.append(np.average(g_loss))
    disc_loss.append(np.average(d_loss))
    epochs_range = range(epoch+1)
    print(epochs_range)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, gen_loss, label='Generator_loss')
    plt.plot(epochs_range, disc_loss, label='Discriminator_loss')
    plt.plot(epochs_range, cls_loss, label='Classifier_loss')
    plt.legend(loc='lower right')
    plt.title('Generator and discriminator loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Test accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    return acc, gen_loss, disc_loss, cls_loss


def Validation(target_val_ds, encoder_t, classifier):
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_accuracy = tf.keras.metrics.CategoricalAccuracy('val_accuracy')
    kappa = tfa.metrics.CohenKappa(num_classes=CLASSES_NUM, sparse_labels=False)
    recall = tf.metrics.Recall(name='recall')
    AA = 0
    for val_batch in target_val_ds.as_numpy_iterator():
        x, y = get_data_from_batch(val_batch)
        x_feature = encoder_t(x, training=False)
        prediction = classifier(x_feature, training=False)
        val_loss(cat_cross_entropy(y, prediction))
        val_accuracy(y, prediction)
        kappa.update_state(y, prediction)
        try:
            recall.update_state(y, prediction)
            AA = recall.result().numpy() / CLASSES_NUM
        except InvalidArgumentError:
            print('预测值不在[0, 1]之间')
    template = 'Loss: {:2f}, OA: {:.2f}%, AA:{:.2f} kappa:{:.2f}'
    print(template.format(val_loss.result(),
                          val_accuracy.result() * 100,
                          AA * 100,
                          kappa.result().numpy() * 100))


def Validation_data(target_val_ds, classifier, encoder=None, flag=False):
    start = time.time()
    loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    oa = tf.keras.metrics.BinaryAccuracy('Overall_accuracy', dtype=tf.float32)
    aa = tf.keras.metrics.CategoricalAccuracy('Average_accuracy')
    kappa = tfa.metrics.CohenKappa(num_classes=CLASSES_NUM, sparse_labels=False)
    m = tf.keras.metrics.Accuracy()
    for val_batch in target_val_ds.as_numpy_iterator():
        x, y = get_data_from_batch(val_batch)
        if flag:
            x = encoder(x, training=False)
        prediction = classifier(x, training=False)
        loss(cat_cross_entropy(y, prediction))
        oa.update_state(y, prediction)
        aa(y, prediction)
        kappa.update_state(y, prediction)
    template = 'Loss: {:2f}, OA: {:.2f}%, AA:{:.2f}%, kappa:{:.2f}'
    print(template.format(loss.result(),
                          oa.result().numpy() * 100,
                          aa.result() * 100,
                          kappa.result().numpy() * 100))
    print('Time for validation is {:.2f} sec'.format(time.time()-start))
    with validation_summary_writer.as_default():
        tf.summary.scalar('loss', loss.result(), step=1)
        tf.summary.scalar('OA', oa.result(), step=1)
        tf.summary.scalar('AA', aa.result(), step=1)
        tf.summary.scalar('Kappa', kappa.result().numpy(), step=1)
