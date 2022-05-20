import os

from train import *
from utils import *
import os
import DCGAN
import tensorflow as tf
import tensorflow_addons as tfa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN_FLAG = 3

NEW_DATA_PATH = './new data'
"""load data"""
source_dict = sio.loadmat(os.path.join(NEW_DATA_PATH, 'Source.mat'))
source_train_ds, source_test_ds = gen_dataset_from_dict(source_dict)
target_dict = sio.loadmat(os.path.join(NEW_DATA_PATH, 'Target.mat'))
target_train_ds, target_test_ds, target_val_ds = gen_dataset_from_dict(target_dict, Val=True)

generator_s = make_generator_model()
generator_t = make_generator_model()
generator_DC = DCGAN.make_generator_model()

discriminator_t = make_discriminator_model()
discriminator_s = make_discriminator_model()
discriminator_DC = DCGAN.make_discriminator_model()
discriminator_domain = make_discriminator_domain_model()
encoder_s = make_encoder_model()
encoder_t = make_encoder_model()
classifier = make_classifier_model()
classifier_data = make_data_classifier_model()

if TRAIN_FLAG == 1:
    train1(classifier_data, target_train_ds, target_test_ds, EPOCHS)
    Validation_data(target_val_ds, classifier_data)
elif TRAIN_FLAG == 2:
    train2(generator_s, discriminator_t, classifier_data,
           source_train_ds, target_train_ds, target_test_ds, EPOCHS)
    Validation_data(target_val_ds, classifier_data)
elif TRAIN_FLAG == 3:
    train3(generator_s, discriminator_t,
           generator_t, discriminator_s,
           classifier_data,
           source_train_ds, target_train_ds, target_test_ds, EPOCHS)
    Validation_data(target_val_ds, classifier_data)
elif TRAIN_FLAG == 4:
    train4(encoder_t, classifier, target_train_ds, target_train_ds, EPOCHS)
    Validation_data(target_val_ds, classifier, encoder=encoder_t, flag=True)
elif TRAIN_FLAG == 5:
    train5(generator_s, discriminator_t,
           generator_t, discriminator_s,
           encoder_s, encoder_t,
           discriminator_domain, classifier,
           source_train_ds, target_train_ds, target_test_ds, EPOCHS)
    Validation_data(target_val_ds, classifier, encoder=encoder_t, flag=True)
elif TRAIN_FLAG == 6:
    train2(generator_DC, discriminator_DC, classifier_data,
           source_train_ds, target_train_ds, target_train_ds, EPOCHS)
    Validation_data(target_val_ds, classifier_data)

plot_result(generator_s, source_train_ds, target_val_ds, TRAIN_FLAG-1)
