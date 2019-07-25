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
import srgan as srgan

from tensorflow.keras.losses import mean_absolute_error, mean_squared_error

from gan_callback import learning_rate
from tensorflow.keras.applications.vgg19 import preprocess_input
from utils import init_session

#set WANDB_API_KEY=9f788fa58123000d5bc285b88c38744a99bdc918
run = wandb.init(project='superres')
config = run.config

config.num_epochs = 1000
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
config.learning_rate = 0.001
config.learning_rate_step_size = 100
config.learning_rate_decay = 0.1
config.discriminator_learning_rate = 0.001
config.label_noise = 0.05

val_dir = 'data/test'
train_dir = 'data/train'
ckpt_dir = "data/ckpt"

filepath = 'data/ckpt/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
#filepath = 'data/ckpt/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-dis{perceptual_distance:.4f}.h5'
image_name = "data/ckpt/validate_image/{0}_ep{1}_it{2}_.jpg"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

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

def cal_perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def content_loss(hr, sr):
    vgg = srgan.vgg_54()
    #sr = preprocess_input(sr)
    #hr = preprocess_input(hr)
    sr_features = vgg(sr)
    hr_features = vgg(hr)
    loss = mean_squared_error(hr_features, sr_features)
    print("content_loss=",loss)
    return loss



class ImageLogger_my(Callback):
    def on_epoch_end(self, epoch, logs):
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
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)





generator = wdsr.wdsr_b(8,32,10,6)
#generator.load_weights("data/ckpt/model-gen-ep263-57.0387.h5")
#generator = edsr.edsr(8,64,8,None,True)
generator_optimizer = Adam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None)


discriminator = srgan.discriminator()
#discriminator.load_weights("data/ckpt/model-des-ep263-57.0387.h5")  
discriminator_optimizer = Adam(lr=config.discriminator_learning_rate)
discriminator.compile(loss='mae',
                        optimizer=discriminator_optimizer,
                        metrics=[])

gan = srgan.srgan(generator, discriminator)
gan.compile(loss=[perceptual_distance, 'mae'],
            loss_weights=[0.06, 1],
            #loss_weights=[0.05, 1],
            optimizer=generator_optimizer)

generator_lr_scheduler = learning_rate(step_size=config.learning_rate_step_size, decay=config.learning_rate_decay, verbose=0)
generator_lr_scheduler.set_model(gan)

discriminator_lr_scheduler = learning_rate(step_size=config.learning_rate_step_size, decay=config.learning_rate_decay, verbose=0)
discriminator_lr_scheduler.set_model(discriminator)

best_val_loss = 0

for epoch in range(config.num_epochs):

    generator_lr_scheduler.on_epoch_begin(epoch)
    discriminator_lr_scheduler.on_epoch_begin(epoch)

    d_losses = []
    g_losses_0 = []
    g_losses_1 = []
    g_losses_2 = []
    distance_mean = []

    gen = image_generator(config.batch_size, train_dir)

    for iteration in range(int(config.steps_per_epoch+1)):

        # ----------------------
        #  Train Discriminator
        # ----------------------

        lr, hr = next(gen)
        sr = generator.predict(lr)

        hr_labels = np.ones(config.batch_size) + config.label_noise * np.random.random(config.batch_size)
        sr_labels = np.zeros(config.batch_size) + config.label_noise * np.random.random(config.batch_size)

        hr_loss = discriminator.train_on_batch(hr, hr_labels)
        sr_loss = discriminator.train_on_batch(sr, sr_labels)

        d_losses.append((hr_loss + sr_loss) / 2)

        # ------------------
        #  Train Generator
        # ------------------

        lr, hr = next(gen)

        labels = np.ones(config.batch_size)

        perceptual_loss = gan.train_on_batch(lr, [hr, labels])

        g_losses_0.append(perceptual_loss[0])
        g_losses_1.append(perceptual_loss[1])
        g_losses_2.append(perceptual_loss[2])


        print(f'[{epoch:03d}-{iteration:03d}] '
                f'discriminator loss = {np.mean(d_losses[:]):.3f}| '
                f'generator loss = {np.mean(g_losses_0[:]):.3f}| '
                f'g_distance = {np.mean(g_losses_1[:]):.3f}| '
                f'gan_d_loss = {np.mean(g_losses_2[:]):.3f}| ')

    generator_lr_scheduler.on_epoch_end(epoch)
    discriminator_lr_scheduler.on_epoch_end(epoch)

    val_perceptual_distance = []

    for i in range(int(config.steps_per_epoch+1)):
        val_generator = image_generator(config.batch_size, val_dir)
        in_sample_images, out_sample_images = next(val_generator)
        sr = generator.predict(in_sample_images)
        dis = cal_perceptual_distance(out_sample_images,sr)
        val_perceptual_distance.append(dis)

    g_loss = np.mean(g_losses_1[:])
    val_loss = np.mean(val_perceptual_distance[:])
    

    wandb.log({'loss':g_loss, 'val_loss':val_loss, 'val_perceptual_distance':val_loss}, '', step=epoch)

    if (best_val_loss > val_loss or best_val_loss==0):
        model_path_g = f'data/ckpt/model-gen-ep{epoch:03d}-{val_loss:.4f}.h5'
        generator.save(model_path_g)
        print(f'{epoch},{np.mean(d_losses)},{np.mean(g_losses_0)}\n')
        model_path_d = f'data/ckpt/model-des-ep{epoch:03d}-{val_loss:.4f}.h5'
        print('Saving model',model_path_g)
        discriminator.save_weights(model_path_d)
        wandb.run.summary["loss"] = g_loss
        wandb.run.summary["val_loss"] = val_loss
        wandb.run.summary["val_perceptual_distance"] = val_loss
        best_val_loss = val_loss

    