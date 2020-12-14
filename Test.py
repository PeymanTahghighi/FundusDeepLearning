# '''
#   A simple Conv3D example with Keras
# '''
# import tensorflow as tf
# import keras
# #from keras.models import Sequential
# #from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
# from keras.utils import to_categorical
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Model
# from DeformableConv3D import *

# # -- Preparatory code --
# # Model configuration
# batch_size = 100
# no_epochs = 30
# learning_rate = 0.001
# no_classes = 10
# validation_split = 0.2
# verbosity = 1

# # Convert 1D vector into 3D values, provided by the 3D MNIST authors at
# # https://www.kaggle.com/daavoo/3d-mnist
# def array_to_color(array, cmap="Oranges"):
#   s_m = plt.cm.ScalarMappable(cmap=cmap)
#   return s_m.to_rgba(array)[:,:-1]

# # Reshape data into format that can be handled by Conv3D layers.
# # Courtesy of Sam Berglin; Zheming Lian; Jiahui Jang - University of Wisconsin-Madison
# # Report - https://github.com/sberglin/Projects-and-Papers/blob/master/3D%20CNN/Report.pdf
# # Code - https://github.com/sberglin/Projects-and-Papers/blob/master/3D%20CNN/network_final_version.ipynb
# def rgb_data_transform(data):
#   data_t = []
#   for i in range(data.shape[0]):
#     data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
#   return np.asarray(data_t, dtype=np.float32)

# # -- Process code --
# # Load the HDF5 data file
# with h5py.File("./full_dataset_vectors.h5", "r") as hf:    

#     # Split the data into training/test features/targets
#     X_train = hf["X_train"][:]
#     targets_train = hf["y_train"][:]
#     X_test = hf["X_test"][:] 
#     targets_test = hf["y_test"][:]

#     # Determine sample shape
#     sample_shape = (16, 16, 16, 3)

#     # Reshape data into 3D format
#     X_train = rgb_data_transform(X_train)
#     X_test = rgb_data_transform(X_test)

#     # Convert target vectors to categorical targets
#     targets_train = to_categorical(targets_train).astype(np.integer)
#     targets_test = to_categorical(targets_test).astype(np.integer)
    
#     # Create the model
#     inp = tf.keras.layers.Input(shape = sample_shape,name="input");
#     initializer = tf.random_normal_initializer(0.0,0.1);
#     x = DeformableConvLayer3D(filters = 16, kernel_size=(3, 3, 3), activation=tf.keras.layers.LeakyReLU(0.2), kernel_initializer=initializer)(inp);
#     x = tf.keras.layers.Conv3D(filters = 256, kernel_size=(3, 3, 3), activation=tf.keras.layers.LeakyReLU(0.2), kernel_initializer=initializer)(x);

#     x = tf.keras.layers.Flatten()(x);
#     x = tf.keras.layers.Dense(no_classes,activation='softmax')(x);

#     model = Model(inputs = inp,outputs = x);

#     total_parameters = 0
#     for variable in model.trainable_variables:
#         # shape is an array of tf.Dimension
#         shape = variable.get_shape()
#         print(shape)
#         print(len(shape))
#         variable_parameters = 1
#         for dim in shape:
#             print(dim)
#             variable_parameters *= dim
#         print(variable_parameters)
#         total_parameters += variable_parameters
#     print(total_parameters)

#     model.summary();

#     # model = tf.keras.layers.Sequential()
#     # model.add(tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
#     # model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
#     # model.add(tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
#     # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#     # model.add(tf.keras.layers.Flatten())
#     # model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
#     # model.add(tf.keras.layers.Dense(no_classes, activation='softmax'))

#     # Compile the model
#     model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                   optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
#                   metrics=['accuracy'])

#     # Fit data to model
#     history = model.fit(X_train, targets_train,
#                 batch_size=Config.BATCH_SIZE,
#                 epochs=no_epochs,
#                 validation_split=validation_split)

#     # Generate generalization metrics
#     score = model.evaluate(X_test, targets_test, verbose=0)
#     print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

#TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras
# from Conv2D import *
# from ConvTranspose2D import *

# # Helper libraries
# import numpy as np
# import matplotlib.pyplot as plt

# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# train_images = train_images / 255.0
# train_images = np.expand_dims(train_images,axis=-1);


# test_images = test_images / 255.0

# initializer = tf.random_normal_initializer(0.0,0.02);
# inp = tf.keras.layers.Input(shape=(28,28,1));
# x = ConvLayer2D(filters = 64,kernel_size=3,kernel_initializer = initializer,deformable=True)(inp);
# #x = DeformableConvTransposeLayer2D(filters = 16,kernel_size=(3,3),kernel_initializer = initializer)(x);
# #x = ConvTransposeLayer2D(filters=32,kernel_size=3,kernel_initializer=initializer,deformable=True)(x);
# x = ConvTransposeLayer2D(filters = 128,kernel_size=3,kernel_initializer = initializer,strides=2)(x);
# x = tf.keras.layers.Flatten()(x);
# x = tf.keras.layers.Dense(10)(x);
# # model = keras.Sequential([
# #     keras.layers.Flatten(input_shape=(28, 28)),
# #     keras.layers.Dense(128, activation='relu'),
# #     keras.layers.Dense(10)
# # ])

# model = tf.keras.models.Model(inputs = inp,outputs  = x);

# model.summary();

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(train_images, train_labels,batch_size = Config.BATCH_SIZE, epochs=10)


import Fetcher
import cv2

data = Fetcher.DataFetcher(imagePath="Dataset/fundus", heightmapPath="Dataset/heightmap");

data.load();

for i in range(data.totalDataSize):
    img,_,_ = data.fetchData(1);
    cv2.imwrite("CLAHE/"+str(i)+"_.jpg",img[0]);


