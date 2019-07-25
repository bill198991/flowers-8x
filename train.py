import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from weightnorm import AdamWithWeightnorm as wn
import wandb
from wandb.keras import WandbCallback
import wdsr as wdsr
import essr as essr

from tensorflow.keras.losses import mean_absolute_error, mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#set WANDB_API_KEY=9f788fa58123000d5bc285b88c38744a99bdc918
run = wandb.init(project='superres')
config = run.config

config.num_epochs = 40000
config.batch_size = 64
#batch_size = 10 #for dbpn
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
#leaning_rete = 1e-3
config.leaning_rete = 0.001

val_dir = 'data/test'
train_dir = 'data/train'
ckpt_dir = "data/ckpt"

#filepath = 'data/ckpt/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
filepath = 'data/ckpt/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-dis{val_perceptual_distance:.4f}.h5'
image_name = "data/ckpt/validate_image/{0}_ep{1}_it{2}_.jpg"

checkpoint = ModelCheckpoint(filepath, monitor='val_perceptual_distance', verbose=1, save_best_only=True, mode='min')

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xzf - -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) / config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) / config.batch_size


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def loss_perceptual_distance(y_true, y_pred):
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)),axis=-1)


val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)


class ImageLogger_my(Callback):
    def on_epoch_end(self, epoch, logs):
        if (epoch % 100 == 0):
            preds = self.model.predict(in_sample_images)
            in_resized = []
            image_data = []
            for arr in in_sample_images:
                # Simple upsampling
                in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))

            for i,o in enumerate(preds):
                if (i%10 == 0):
                    image_data = np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)
                    input_image = Image.fromarray(np.uint8(image_data))
                    input_image.save(image_name.format("result",epoch,i))

class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        if (epoch % 100 == 0):
            preds = self.model.predict(in_sample_images)
            in_resized = []
            for arr in in_sample_images:
                # Simple upsampling
                in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
            wandb.log({
                "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
            }, commit=False)


"""
model = Sequential()
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same',
                        input_shape=(config.input_width, config.input_height, 3)))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
loss = mae
"""

#model = load_model('model.h5'，{'perceptual_distance': perceptual_distance})


#model = wdsr.wdsr_b(8,48,6,6)
model = essr.essr()
#loss = loss_perceptual_distance
loss = mean_squared_error
optimizer_wdsr = Adam(lr=config.leaning_rete, beta_1=0.9, beta_2=0.999, epsilon=None)


#model = edsr.edsr_generator(8,64,32)
#loss = mean_squared_error


#model = dbpn.dbpn()
#model.load_weights("data/ckpt/model-ep107-loss56.864-val_loss57.801.h5")  
#loss = mean_squared_error
#loss = loss_perceptual_distance
#optimizer_dbpn = Adam(lr=leaning_rete, beta_1=0.9, beta_2=0.999, epsilon=None)

model.compile(optimizer=optimizer_wdsr, loss=loss,metrics=[perceptual_distance])
#model.compile(optimizer=optimizer_dbpn, loss=loss,metrics=[perceptual_distance])
#model.compile(optimizer=wn(lr=leaning_rete), loss=loss,
#              metrics=[perceptual_distance])

model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs,
                    callbacks = [checkpoint,ImageLogger(),WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)
