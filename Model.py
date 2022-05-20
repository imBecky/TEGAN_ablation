import time
from IPython import display
from utils import *
from ResNet import *
import ResNet
import tensorflow_addons as tfa

BANDS = 72
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(72, 1)))
    model.add(ResNet.ResBlock_up_top(1))
    model.add(tf.keras.layers.Reshape(target_shape=(72, 2)))
    model.add(tf.keras.layers.Dropout(0.7))
    model.add(tf.keras.layers.Dense(1, use_bias=False))
    model.build()
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(72, 1)))
    model.add(ResNet.ResBlock_Down(1))
    model.add(layers.Dropout(0.7))
    model.add(layers.Flatten())
    model.add(ResNet.Res_Dense(1))
    return model


def make_discriminator_domain_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(FEATURE_dim, 1)))
    model.add(ResNet.ResBlock_Down(1))
    model.add(layers.Dropout(0.7))
    model.add(layers.Flatten())
    model.add(ResNet.Res_Dense(1))
    return model


"""This classifier is for raw data classification"""


def make_data_classifier_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(FEATURE_dim, input_shape=(BANDS, 1)))
    model.add(layers.Conv1D(filters=1,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            use_bias=False))
    model.add(layers.Dropout(0.9))
    model.add(layers.Flatten())
    model.add(layers.Dense(CLASSES_NUM))
    return model


"""This classifier is for encoded feature"""


def make_classifier_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(FEATURE_dim, input_shape=(36, 1)))
    model.add(ResNet.ResBlock_up_top(CLASSES_NUM))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(CLASSES_NUM))
    return model


def make_encoder_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(72, 1)))
    model.add(layers.Conv1D(filters=72,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            use_bias=False))
    model.add(layers.MaxPool1D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1))
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return binary_cross_entropy(tf.ones_like(fake_output), fake_output)


def classifier_loss(prediction, label):
    return cat_cross_entropy(label, prediction)


generator_s_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
generator_t_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
discriminator_t_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
discriminator_s_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
discriminator_domain_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
encoder_s_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
encoder_t_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
classifier_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
classifier_data_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
