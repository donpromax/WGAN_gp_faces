import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Generator(keras.Model):
    def __init__(self, is_training=True):
        super(Generator, self).__init__()
        # z: [b, 100] => [b, 64, 64, 3]
        # channel decrease, image size increase
        self.fc = layers.Dense(3 * 3 * 512)

        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)
        #
        x = self.bn1(self.conv1(x), training=training)
        x = tf.nn.leaky_relu(x)

        x = self.bn2(self.conv2(x), training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv3(x)
        x = tf.tanh(x)

        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b, 64, 64, 3] => [b, 1]
        self.conv1 = layers.Conv2D(64, 5, 4, 'valid')

        self.conv2 = layers.Conv2D(128, 5, 4, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 4, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(512, 5, 4, 'valid')
        self.bn4 = layers.BatchNormalization()

        self.flatten = layers.Flatten()

        self.fc = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        # [b, h, w, c] => [b, -1]
        x = self.flatten(x)
        print(x.shape)
        logits = self.fc(x)
        return logits


def main():
    img_dim = 96
    z_dim = 100
    num_layers = int(np.log2(img_dim)) - 3
    max_num_channels = img_dim * 8
    f_size = img_dim // 2 ** (num_layers + 1)
    batch_size = 256

    print(num_layers)
    print(max_num_channels)
    print(f_size)

    d = Discriminator()
    g = Generator()

    x = tf.random.normal([2, 96, 96, 3])
    z = tf.random.normal([2, 100])

    prob = d(x)
    print(prob.shape)

    # pic = g(z)
    # print(pic.shape)


if __name__ == '__main__':
    main()
