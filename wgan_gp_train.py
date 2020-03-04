import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from PIL import Image
import glob
from gan import Generator, Discriminator
from tensorflow import keras
from dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate(
                (single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def gradient_penalty(discriminator, real_image, fake_image):
    batchsz = real_image.shape[0]
    # dtype caused disconvergence?
    t = tf.random.uniform([batchsz, 1, 1, 1], minval=0.,
                          maxval=1., dtype=tf.float32)
    x_hat = t * real_image + (1. - t) * fake_image
    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        Dx = discriminator(x_hat, training=True)
    grads = tape.gradient(Dx, x_hat)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp


def d_loss_fn(generator, discriminator, batch_z, real_image):
    fake_image = generator(batch_z, training=True)
    d_fake_score = discriminator(fake_image, training=True)
    d_real_score = discriminator(real_image, training=True)

    loss = tf.reduce_mean(d_fake_score - d_real_score)

    gp = gradient_penalty(discriminator, real_image, fake_image) * 10.

    loss = loss + gp
    return loss, gp


def g_loss_fn(generator, discriminator, batch_z):
    fake_image = generator(batch_z, training=True)
    d_fake_logits = discriminator(fake_image, training=True)
    # loss = celoss_ones(d_fake_logits)
    loss = -tf.reduce_mean(d_fake_logits)
    return loss


def main():
    tf.random.set_seed(233)
    np.random.seed(233)

    z_dim = 100
    epochs = 3000000
    batch_size = 512
    learning_rate = 2e-4
    # ratios = D steps:G steps
    ratios = 2

    img_path = glob.glob(os.path.join('faces', '*.jpg'))
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    # generator.load_weights(os.path.join('checkpoints', 'generator-5000'))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))
    # discriminator.load_weights(os.path.join('checkpoints', 'discriminator-5000'))

    g_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
    # a fixed noise for sampling
    z_sample = tf.random.normal([100, z_dim])

    g_loss_meter = keras.metrics.Mean()
    d_loss_meter = keras.metrics.Mean()
    gp_meter = keras.metrics.Mean()

    for epoch in range(epochs):

        # train D
        for step in range(ratios):
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)
            with tf.GradientTape() as tape:
                d_loss, gp = d_loss_fn(
                    generator, discriminator, batch_z, batch_x)

            d_loss_meter.update_state(d_loss)
            gp_meter.update_state(gp)

            gradients = tape.gradient(
                d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(
                zip(gradients, discriminator.trainable_variables))

        # train G
        batch_z = tf.random.normal([batch_size, z_dim])
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z)

        g_loss_meter.update_state(g_loss)

        gradients = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(
            zip(gradients, generator.trainable_variables))

        if epoch % 100 == 0:

            fake_image = generator(z_sample, training=False)

            print(epoch, 'd-loss:', d_loss_meter.result().numpy(),
                  'g-loss', g_loss_meter.result().numpy(),
                  'gp', gp_meter.result().numpy())

            d_loss_meter.reset_states()
            g_loss_meter.reset_states()
            gp_meter.reset_states()

            # save generated image samples
            img_path = os.path.join('images_wgan_gp', 'wgan_gp-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

        if epoch + 1 % 2000 == 0:
            generator.save_weights(os.path.join(
                'checkpoints_gp', 'generator-%d' % epoch))
            discriminator.save_weights(os.path.join(
                'checkpoints_gp', 'discriminator-%d' % epoch))


if __name__ == '__main__':
    main()
