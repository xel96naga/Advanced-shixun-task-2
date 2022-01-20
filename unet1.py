import tensorflow as tf

from absl import flags
from absl import app

import os
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator



# UNET for ISBI-2012 dataset
class Dataset(tf.keras.Model):
    def __init__(self, num_classes):
        super(Dataset, self).__init__()

        # Input
        inputs = tf.keras.layers.Input((512, 512, 1))

        # Contracting part
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        assert conv1.shape[1:] == (512, 512, 64)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        assert pool1.shape[1:] == (256, 256, 64)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        assert conv2.shape[1:] == (256, 256, 128)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        assert pool2.shape[1:] == (128, 128, 128)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        assert conv3.shape[1:] == (128, 128, 256)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        assert pool3.shape[1:] == (64, 64, 256)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = tf.keras.layers.Dropout(0.5)(conv4)
        assert drop4.shape[1:] == (64, 64, 512)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        assert pool4.shape[1:] == (32, 32, 512)

        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            pool4)
        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            conv5)
        assert conv5.shape[1:] == (32, 32, 1024)
        drop5 = tf.keras.layers.Dropout(0.5)(conv5)

        # Expansive part
        up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
        assert up6.shape[1:] == (64, 64, 512)
        merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
        assert merge6.shape[1:] == (64, 64, 1024)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge6)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        assert conv6.shape[1:] == (64, 64, 512)

        up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
        assert up7.shape[1:] == (128, 128, 256)
        merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
        assert merge7.shape[1:] == (128, 128, 512)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge7)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        assert conv7.shape[1:] == (128, 128, 256)

        up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
        assert up8.shape[1:] == (256, 256, 128)
        merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
        assert merge8.shape[1:] == (256, 256, 256)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        assert conv8.shape[1:] == (256, 256, 128)

        up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
        assert up9.shape[1:] == (512, 512, 64)
        merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
        assert merge9.shape[1:] == (512, 512, 128)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        assert conv9.shape[1:] == (512, 512, 64)
        conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        assert conv9.shape[1:] == (512, 512, 2)
        conv10 = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(conv9)
        assert conv10.shape[1:] == (512, 512, num_classes)

        model = tf.keras.Model(inputs=inputs, outputs=conv10)

        self.model = model

        # print model structure
        self.model.summary()

    def call(self, x):
        return self.model(x)


# binary cross entropy for ISBI-2012 dataset
binary_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# set seed
tf.random.set_seed(1234)

flags.DEFINE_string('checkpoint_path', default='saved_model_isbi_2012/unet_model.h5',
                    help='path to a directory to save model checkpoints during training')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log_isbi_2012',
                    help='path to a directory to save tensorboard log')
flags.DEFINE_integer('num_epochs', default=3, help='training epochs')
flags.DEFINE_integer('steps_per_epoch', default=100, help='steps per epoch')
flags.DEFINE_integer('num_classes', default=1, help='number of prediction classes')

FLAGS = flags.FLAGS

# set configuration value
batch_size = 2
learning_rate = 0.0001


# normalize isbi-2012 data
def normolize(input_images, mask_labels):
    # 0~255 -> 0.0~1.0
    input_images = input_images / 255
    mask_labels = mask_labels / 255

    # set label to binary
    mask_labels[mask_labels > 0.5] = 1
    mask_labels[mask_labels <= 0.5] = 0

    return input_images, mask_labels


# make data generator
def make_train_generator(batch_size, aug_dict):
    image_gen = ImageDataGenerator(**aug_dict)
    mask_gen = ImageDataGenerator(**aug_dict)

    # set image and mask same augmentation using same seed
    image_generator = image_gen.flow_from_directory(
        directory='./isbi_2012/preprocessed',
        classes=['train_imgs'],
        class_mode=None,
        target_size=(512, 512),
        batch_size=batch_size,
        color_mode='grayscale',
        seed=1
    )
    mask_generator = mask_gen.flow_from_directory(
        directory='./isbi_2012/preprocessed',
        classes=['train_labels'],
        class_mode=None,
        target_size=(512, 512),
        batch_size=batch_size,
        color_mode='grayscale',
        seed=1
    )
    train_generator = zip(image_generator, mask_generator)
    for (batch_images, batch_labels) in train_generator:
        batch_images, batch_labels = normolize(batch_images, batch_labels)

        yield (batch_images, batch_labels)


# show image
def show(show_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(show_list)):
        plt.subplot(1, len(show_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(show_list[i]))
        plt.axis('off')
    plt.show()


# show image and save
def show_and_save(show_list, epoch):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(show_list)):
        plt.subplot(1, len(show_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(show_list[i]))
        plt.axis('off')
    #plt.savefig(f'epoch {epoch}.jpg')


# make prediction mask
def create_mask(pred_mask):
    pred_mask = np.where(pred_mask > 0.5, 1, 0)

    return pred_mask[0]


# show prediction
def show_predictions(model, sample_image, sample_mask):
    show([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# show and save prediction
def save_predictions(epoch, model, sample_image, sample_mask):
    show_and_save([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))], epoch)



def main(_):
    # set augmentation
    aug_dict = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    # make generator
    train_generator = make_train_generator(batch_size, aug_dict)

    # data sanity check
    for iter, batch_data in enumerate(train_generator):
        if iter >= 2:  # manually detect the end of the epoch
            break
        batch_image, batch_mask = batch_data[0], batch_data[1]
        sample_image, sample_mask = batch_image[0], batch_mask[0]

    # show data
    show([sample_image, sample_mask])

    # create ISBI model
    unet_model = Dataset(FLAGS.num_classes)

    # show prediction before training
    show_predictions(unet_model, sample_image, sample_mask)

    # set optimizer
    optimizer = tf.optimizers.Adam(learning_rate)

    # check if checkpoint path exists
    if not os.path.exists(FLAGS.checkpoint_path.split('/')[0]):
        os.mkdir(FLAGS.checkpoint_path.split('/')[0])

    # restore latest checkpoint
    if os.path.isfile(FLAGS.checkpoint_path):
        unet_model.load_weights(FLAGS.checkpoint_path)
        print(f'{FLAGS.checkpoint_path} checkpoint is restored!')

    # set compile
    unet_model.compile(optimizer=optimizer, loss=binary_loss_object, metrics=['accuracy'])

    # start training
    unet_model.fit_generator(train_generator,
                             steps_per_epoch=FLAGS.steps_per_epoch,
                             epochs=FLAGS.num_epochs)



if __name__ == '__main__':
    app.run(main)