import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if(len(physical_devices)>0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


base_model = tf.keras.applications.InceptionV3(weights= None,
                                               input_shape= (150,150,3),
                                               include_top= False)

local_weights_file = "./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
base_model.load_weights(local_weights_file)

for layer in base_model.layers:
    layer.trainable = False

base_model.summary()

last_layer = base_model.get_layer("mixed7")
last_layer_output = last_layer.output
print(last_layer_output.shape)

x = tf.keras.layers.Flatten()(last_layer_output)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation= "sigmoid")(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = 0.0001, momentum= 0.9),
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

train_dir = "./temp/training/"
validation_dir = "./temp/testing/"

train_datagen = ImageDataGenerator(rescale = 1./255.0,
                                   rotation_range=40,
                                   height_shift_range=0.2,
                                   width_shift_range= 0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255.0)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size= 32,
                                                    target_size=(150,150),
                                                    class_mode="binary")

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=32,
                                                              target_size=(150,150),
                                                              class_mode="binary")

whole_train_dir = "./temp/WholeTrain/"

whole_train_datagen = ImageDataGenerator(rescale= 1./255.0,
                                         rotation_range=40,
                                         height_shift_range=0.2,
                                         width_shift_range=0.2,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)

whole_train_generator = whole_train_datagen.flow_from_directory(whole_train_dir,
                                                                batch_size=32,
                                                                target_size=(150,150),
                                                                class_mode="binary")


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get("accuracy")>0.97):
            self.model.stop_training = True

root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

callback = myCallback()

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

st = time.time()

history = model.fit(train_generator, epochs = 50, steps_per_epoch=len(train_generator),
                    validation_data=validation_generator,
                    callbacks=[callback,tensorboard_cb, tf.keras.callbacks.EarlyStopping(patience = 10)], verbose = 1)

print("Total time it took is: {}".format((time.time()- st)/60))