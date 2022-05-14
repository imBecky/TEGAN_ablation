import datetime

import scipy.io as sio
from Model import *
from utils import *
from test import *
import os

loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
oa = tf.keras.metrics.CategoricalAccuracy('Overall accuracy')
aa = tf.keras.metrics.CategoricalAccuracy('Average_accuracy')
kappa = tfa.metrics.CohenKappa(num_classes=CLASSES_NUM, sparse_labels=False, name='Kappa')


@tf.function
def train_step_1(classifier_data,
                 target_batch):
    """classify the raw data,
       =========BLOCK 1=========="""
    data, label = get_data_from_batch(target_batch)
    with tf.GradientTape() as tape:
        prediction = classifier_data(data, training=True)
        classify_loss = classifier_loss(prediction, label)

        gradient = tape.gradient(classify_loss,
                                 classifier_data.trainable_variables)
        classifier_data_optimizer.apply_gradients(zip(gradient,
                                                      classifier_data.trainable_variables))
    loss(classify_loss)
    oa(label, prediction)
    aa(label, prediction)
    kappa.update_state(label, prediction)


def train1(classifier_data, target_train_ds, target_test_ds, epochs):
    best = 0
    patience = PATIENCE
    wait = 0
    for epoch in range(epochs):
        start = time.time()
        for target_batch in target_train_ds.as_numpy_iterator():
            train_step_1(classifier_data, target_batch)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('oa', oa.result(), step=epoch)
            tf.summary.scalar('aa', aa.result(), step=epoch)
            tf.summary.scalar('kappa', kappa.result(), step=epoch)
        print(f'1:Time for epoch {epoch + 1} is {time.time() - start} sec')
        for target_batch in target_test_ds.as_numpy_iterator():
            test_step1(classifier_data, target_batch)
        now = test_acc.result().numpy()
        if epoch > int(epochs/2):
            wait += 1
            if now > best:
                best = now
                wait = 0
            if wait >= patience:
                break


@tf.function
def train_step2_1(generator_s, discriminator_t,
                  source_batch, target_batch):
    x_s, y_s = get_data_from_batch(source_batch)
    x_t, y_t = get_data_from_batch(target_batch)
    with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
        x_t_fake = generator_s(x_s, training=True)
        real_output = discriminator_t(x_t, training=True)
        fake_output = discriminator_t(x_t_fake, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)

        gen_gradient = tape_g.gradient(gen_loss, generator_s.trainable_variables)
        disc_gradient = tape_d.gradient(disc_loss, discriminator_t.trainable_variables)

        generator_s_optimizer.apply_gradients(zip(gen_gradient, generator_s.trainable_variables))
        discriminator_t_optimizer.apply_gradients(zip(disc_gradient, discriminator_t.trainable_variables))


@tf.function
def train_step_2_2(generator_s, classifier_data,
                   source_batch, target_batch):
    x_s, y_s = get_data_from_batch(source_batch)
    x_t, y_t = get_data_from_batch(target_batch)
    with tf.GradientTape() as tape:
        x_t_fake = generator_s(x_s, training=False)
        prediction_s = classifier_data(x_t_fake, training=True)
        prediction_t = classifier_data(x_t, training=True)

        loss1 = classifier_loss(prediction_s, y_s)
        loss1 += classifier_loss(prediction_t, y_t)

        gradient = tape.gradient(loss1, classifier_data.trainable_variables)
        classifier_data_optimizer.apply_gradients(zip(gradient, classifier_data.trainable_variables))
    loss(loss1)
    oa(y_t, prediction_t)
    aa(y_t, prediction_t)
    kappa.update_state(y_t, prediction_t)


def train2(generator_s, discriminator_t, classifier_data,
           source_ds, target_ds, epochs):
    best = 0
    patience = PATIENCE
    wait = 0
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                train_step2_1(generator_s, discriminator_t,
                              source_batch, target_batch)
        print(f'2:Time for epoch {epoch + 1} is {time.time() - start} sec')
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                train_step_2_2(generator_s, classifier_data,
                               source_batch, target_batch)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('oa', oa.result(), step=epoch)
            tf.summary.scalar('aa', aa.result(), step=epoch)
            tf.summary.scalar('kappa', kappa.result(), step=epoch)
        print(f'2:Time for epoch {epoch + 1} is {time.time() - start} sec')


@tf.function
def train_step3_1(generator_s, discriminator_t,
                  generator_t, discriminator_s,
                  source_batch, target_batch):
    """cycleGAN block"""
    x_s, y_s = get_data_from_batch(source_batch)
    x_t, y_t = get_data_from_batch(target_batch)
    with tf.GradientTape() as tape_gs, tf.GradientTape() as tape_dt, \
            tf.GradientTape() as tape_gt, tf.GradientTape() as tape_ds:
        if x_s.shape[0] > x_t.shape[0]:
            x_s = x_t[:x_t.shape[0]]
        elif x_s.shape[0] < x_t.shape[0]:
            x_t = x_s[:x_s.shape[0]]
        x_t_fake = generator_s(x_s, training=True)
        real_output_t = discriminator_t(x_t, training=True)
        fake_output_t = discriminator_t(x_t_fake, training=True)
        x_s_fake = generator_t(x_t, training=True)
        real_output_s = discriminator_s(x_s, training=True)
        fake_output_s = discriminator_s(x_s_fake, training=True)

        x_s_cyc = generator_t(x_t_fake, training=True)
        x_t_cyc = generator_s(x_s_fake, training=True)
        cycle_consistence_loss = cat_cross_entropy(x_s, x_s_cyc) + cat_cross_entropy(x_t, x_t_cyc)

        disc_t_loss = discriminator_loss(real_output_t, fake_output_t)
        gen_s_loss = generator_loss(fake_output_t)
        gen_s_loss += cycle_consistence_loss

        disc_s_loss = discriminator_loss(real_output_s, fake_output_s)
        gen_t_loss = generator_loss(fake_output_s)
        gen_t_loss += cycle_consistence_loss

        gen_s_gradient = tape_gs.gradient(gen_s_loss, generator_s.trainable_variables)
        disc_t_gradient = tape_dt.gradient(disc_t_loss, discriminator_t.trainable_variables)

        gen_t_gradient = tape_gt.gradient(gen_t_loss, generator_t.trainable_variables)
        disc_s_gradient = tape_ds.gradient(disc_s_loss, discriminator_s.trainable_variables)

        generator_s_optimizer.apply_gradients(zip(gen_s_gradient, generator_s.trainable_variables))
        discriminator_t_optimizer.apply_gradients(zip(disc_t_gradient, discriminator_t.trainable_variables))
        generator_t_optimizer.apply_gradients(zip(gen_t_gradient, generator_s.trainable_variables))
        discriminator_s_optimizer.apply_gradients(zip(disc_s_gradient, discriminator_t.trainable_variables))


def train3(generator_s, discriminator_t,
           generator_t, discriminator_s,
           classifier_data,
           source_ds, target_ds, epochs):
    for epoch in range(600):
        start = time.time()
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                train_step3_1(generator_s, discriminator_t,
                              generator_t, discriminator_s,
                              source_batch, target_batch)
        print(f'3:Time for epoch {epoch + 1} is {time.time() - start} sec')
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                train_step_2_2(generator_s, classifier_data,
                               source_batch, target_batch)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('oa', oa.result(), step=epoch)
            tf.summary.scalar('aa', aa.result(), step=epoch)
            tf.summary.scalar('kappa', kappa.result(), step=epoch)
        print(f'3:Time for epoch {epoch + 1} is {time.time() - start} sec')


@tf.function
def train_step_4(encoder, classifier,
                 target_batch):
    """classify the encoded data
       =========BLOCK 1=========="""
    data, label = get_data_from_batch(target_batch)
    with tf.GradientTape() as tape:
        feature = encoder(data, training=True)
        prediction = classifier(feature, training=True)
        classify_loss = classifier_loss(prediction, label)

        gradient = tape.gradient(classify_loss,
                                 classifier.trainable_variables)
        classifier_optimizer.apply_gradients(zip(gradient,
                                                 classifier.trainable_variables))
    loss(classify_loss)
    oa(label, prediction)
    aa(label, prediction)
    kappa.update_state(label, prediction)


def train4(encoder, classifier,
           target_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        for target_batch in target_ds.as_numpy_iterator():
            train_step_4(encoder, classifier, target_batch)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('oa', oa.result(), step=epoch)
            tf.summary.scalar('aa', aa.result(), step=epoch)
            tf.summary.scalar('kappa', kappa.result(), step=epoch)
        print(f'4:Time for epoch {epoch + 1} is {time.time() - start} sec')


def train_step5_1(generator_s, generator_t,
                  encoder_s, encoder_t,
                  discriminator_domain, classifier,
                  source_batch, target_batch):
    x_s, y_s = get_data_from_batch(source_batch)
    x_t, y_t = get_data_from_batch(target_batch)
    with tf.GradientTape(persistent=True) as tape:
        x_t_fake = generator_s(x_s, training=False)
        x_s_fake = generator_t(x_t, training=False)
        f_s_real = encoder_s(x_s, training=True)
        f_t_real = encoder_t(x_t, training=True)
        f_s_fake = encoder_s(x_s_fake, training=True)
        f_t_fake = encoder_t(x_t_fake, training=False)
        s_real_output = discriminator_domain(f_s_real, training=True)
        t_real_output = discriminator_domain(f_t_real, training=True)
        s_fake_output = discriminator_domain(f_s_fake, training=True)
        t_fake_output = discriminator_domain(f_t_fake, training=True)
        s_real_pred = classifier(f_s_real, training=True)
        t_real_pred = classifier(f_t_real, training=True)
        s_fake_pred = classifier(f_s_fake, training=True)
        t_fake_pred = classifier(f_t_fake, training=True)

        enc_s_loss = binary_cross_entropy(tf.zeros_like(s_real_output), s_real_output)
        enc_s_loss += binary_cross_entropy(tf.zeros_like(s_fake_output), s_fake_output)  # 注：此处可以有平衡因子！！！！！
        enc_t_loss = binary_cross_entropy(tf.zeros_like(t_real_output), t_real_output)
        enc_t_loss += binary_cross_entropy(tf.zeros_like(t_fake_output), t_fake_output)
        disc_loss = discriminator_loss(s_real_output, t_real_output)
        disc_loss += discriminator_loss(s_fake_output, t_fake_output)
        enc_s_loss += cat_cross_entropy(y_s, s_real_pred)
        enc_s_loss += cat_cross_entropy(y_t, s_fake_pred)
        enc_t_loss += cat_cross_entropy(y_t, t_real_pred)
        enc_t_loss += cat_cross_entropy(y_s, t_fake_pred)
        cls_loss = cat_cross_entropy(y_s, s_real_pred)
        cls_loss += cat_cross_entropy(y_t, s_fake_pred)
        cls_loss += cat_cross_entropy(y_t, t_real_pred)
        cls_loss += cat_cross_entropy(y_s, t_fake_pred)

        enc_s_gradient = tape.gradient(enc_s_loss, encoder_s.trainable_variables)
        enc_t_gradient = tape.gradient(enc_t_loss, encoder_t.trainable_variables)
        dic_domain_gradient = tape.gradient(disc_loss, discriminator_domain.trainable_variables)
        cls_gradient = tape.gradient(cls_loss, classifier.trainable_variables)

        encoder_s_optimizer.apply_gradients(zip(enc_s_gradient, encoder_s.trainable_variables))
        encoder_t_optimizer.apply_gradients(zip(enc_t_gradient, encoder_t.trainable_variables))
        discriminator_domain_optimizer.apply_gradients(
            zip(dic_domain_gradient, discriminator_domain.trainable_variables))
        classifier_optimizer.apply_gradients(zip(cls_gradient, classifier.trainable_variables))
    loss(cls_loss)
    oa(y_t, t_real_pred)
    aa(y_t, t_real_pred)
    kappa.update_state(y_t, t_real_pred)
    del tape


def train5(generator_s, discriminator_t,
           generator_t, discriminator_s,
           encoder_s, encoder_t,
           discriminator_domain, classifier,
           source_ds, target_ds,
           target_test_ds,
           epochs):
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                train_step3_1(generator_s, discriminator_t,
                              generator_t, discriminator_s,
                              source_batch, target_batch)
        print(f'5:Time for epoch {epoch + 1} is {time.time() - start} sec')
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                train_step5_1(generator_s, generator_t,
                              encoder_s, encoder_t,
                              discriminator_domain, classifier,
                              source_batch, target_batch)
        print(f'5:Time for epoch {epoch + 1} is {time.time() - start} sec')
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('oa', oa.result(), step=epoch)
            tf.summary.scalar('aa', aa.result(), step=epoch)
            tf.summary.scalar('kappa', kappa.result(), step=epoch)
        # ===========TEST=============
        now = 0
        wait = 0
        best = 0
        patience = PATIENCE
        if epoch % 5 == 0:
            for target_batch in target_test_ds.as_numpy_iterator():
                test_step4(encoder_t, classifier, target_batch)
            template = 'Epoch {}: Test loss={:.2f}, ' \
                       ' test_accuracy={:.2f}%'
            print(template.format(epoch + 1, test_loss.result(),
                                  test_acc.result() * 100))
            with test_summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
                tf.summary.scalar('test_accuracy', test_acc.result(), step=epoch)
            now = test_acc.result().numpy()
        if epoch > 50:
            wait += 1
            if now > best:
                best = now
                wait = 0
            if wait >= patience:
                break

