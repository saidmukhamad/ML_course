import matplotlib.pyplot as plt
import minesweeper
import numpy as np
import random
import tensorflow as tf
import tf_utils
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

num_threads = 10

from absl import app, flags
from tensorflow.keras import layers, losses, models, regularizers


FLAGS = flags.FLAGS

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "10"
os.environ["TF_NUM_INTEROP_THREADS"] = "10"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

flags.DEFINE_integer('width', 32,
                     'The width of the Minesweeper field.',
                     lower_bound=0)
flags.DEFINE_integer('height', 16,
                     'The height of the Minesweeper field.',
                     lower_bound=0)
flags.DEFINE_integer('num_mines', 99,
                     'The number of mines in the field.',
                     lower_bound=0)

flags.DEFINE_string('output_directory', None,
                    'Where to write the output model.')

def create_random_field(width: int,
                        height: int,
                        num_mines: int) -> minesweeper.Field:
    """Create a random field with some areas revealed."""
    field = minesweeper.Field(width, height, num_mines)

    # Sweep a random number of safe cells.
    num_sweeps = random.randrange(5, 25)
    for _ in range(num_sweeps):
        (x, y) = field.RandomSafeCell()
        field.Sweep(x, y)
        if field.IsCompleted():
            break
    return field

def create_probability_tensor(field: minesweeper.Field) -> tf.Tensor:
    """Returns a tensor containing probabilities of `field` containing mines."""
    tensor = np.zeros((field.height, field.width), np.float32)

    for i, row in enumerate(field.proximity):
        for j, value in enumerate(row):
            if value == -1:
                tensor[i][j] = 1.0

    tensor = tensor.reshape(field.height, field.width, 1)
    return tf.convert_to_tensor(tensor, dtype=tf.float32)

def create_examples(width: int,
                    height: int,
                    num_mines: int,
                    num_examples: int) -> (tf.Tensor, tf.Tensor):
    """Returns a Tensorflow dataset containing `num_examples`.

    Returns:
      A (input, output) tuple.
    """
    input_tensors = []
    output_tensors = []
    for _ in range(num_examples):
        field = create_random_field(width, height, num_mines)

        input_tensors.append(tf_utils.create_input_tensor(field))
        output_tensors.append(create_probability_tensor(field))
    return (tf.stack(input_tensors, axis=0), tf.stack(output_tensors, axis=0))

def main(argv):

    width = FLAGS.width
    height = FLAGS.height
    num_mines = FLAGS.num_mines
    output_directory = FLAGS.output_directory

    (train_inputs, train_outputs) = create_examples(width, height, num_mines, 50096)
    (test_inputs, test_outputs) = create_examples(width, height, num_mines, 5012)
    input = layers.Input(shape=(height, width, 11))


    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.5)(conv5)
    
    up6 = layers.concatenate([layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.concatenate([layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    model = layers.BatchNormalization()(conv8)
    model = layers.Dropout(0.5)(conv8)

    up9 = layers.concatenate([layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    # model = tf.keras.Sequential()(input)

    # model = layers.Conv2D(32, (7, 7),
    #                       activation='softmax',
    #                       padding='same')(model)
  
    # model = layers.Conv2D(64, (3, 3),
    #                       activation='softmax',
    #                       padding='same')(model)

    # # model = layers.MaxPooling2D(pool_size=(2,2), padding="same")(model)

    # model = layers.Conv2D(128, (1, 1),
    #                       activation='softmax',
    #                       padding='same')(model)

                          
    # # model = layers.MaxPooling2D(pool_size=(2,2), padding="valid")(model)
    
    # model = layers.Conv2D(128, (3, 3),
    #                       activation='softmax',
    #                       padding='same')(model)


    # model = layers.Conv2D(128, (3, 3),
    #                       activation='softmax',
    #                       padding='same')(model)

    # # model = layers.MaxPooling2D(pool_size=(2,2), padding="valid")(model)

    # # model = layers.BatchNormalization()(model)

    # # flatten = layers.Flatten(name='flatten')(model)
    # # output = layers.Dense(11, activation='relu', name='output')(flatten)


    # model = layers.Conv2D(1, 1, padding='same')(model)
    # # model = layers.Flatten()(model)
    # # model =layers.Dense(64, activation='sigmoid')(model)
    # # output = layers.Dense(5)(model)

    model = models.Model(inputs=input, outputs=conv10)

    model.summary()

    model.compile(optimizer='adam',
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.compile(optimizer='adam',
    #               loss=losses.MeanSquaredError(),
    #               metrics=['accuracy'])

    csvLogger =  tf.keras.callbacks.CSVLogger('epoch3.csv')

    history = model.fit(train_inputs, train_outputs, epochs=100, callbacks=[csvLogger],
                        validation_data=(test_inputs, test_outputs))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    _, test_acc = model.evaluate(test_inputs,  test_outputs, verbose=2)

    print('Accuracy', test_acc)

    if output_directory:
        model.save(output_directory)

if __name__ == "__main__":
    app.run(main)
