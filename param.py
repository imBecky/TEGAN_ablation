import matplotlib as mpl
import tensorflow as tf
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASSES_NUM = 6  # 输出7类地物
LABELS = ['', 'Trees', 'Asphalt', 'Parking lot', 'Bitumen', 'Meadow', 'Soil']
VAL_FRAC = 0.8
TEST_FRAC = 0.9  # target用来测试数据的百分比 test/train
TRAIN_FRAC = 0.1
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
BUFFER_SIZE = 60000
BATCH_SIZE = 47
BANDS = 72
EPOCHS = 650
PATIENCE = 5
noise_dim = 72
num_examples_to_generate = 16
# seed = tf.random.normal([BATCH_SIZE, 72, 1])
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
FEATURE_dim = 36
lr = 1e-4

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/1/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/1/' + current_time + '/test'
val_log_dir = 'logs/gradient_tape/1/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
validation_summary_writer = tf.summary.create_file_writer(val_log_dir)

