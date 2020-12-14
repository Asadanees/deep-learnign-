#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  minst_impover.py
#  
#  Copyright 2020 Asad Anees <asad@asad-Latitude-E7440>
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
##       Improving Computer Vision Accuracy using Convolutions and pooling layers (CNN) MNIST DATA SET
#						total images= 70,000
#						training images= 60,000 and testing images= 10,000
#In the project you could see how you would improve Fashion MNIST using Convolutions. 
# We could improve MNIST to 99.00% accuracy or more using only a single convolutional layer and a single MaxPooling 2D.
# We must stop training once the accuracy goes above our desired limit. 
# It could happen in less than 20 epochs or more, so it's ok to hard code the number of epochs for training, 
# but our training must end once it hits the above metric. If it doesn't, then we'll need to redesign our Convolutions layers.
#Improving Computer Vision Accuracy using Convolutions
#Important note: this code is tested on  Google Colab. Might be you need little modification to complicate and run on your local machine.
#	Import the TensorFlow

import tensorflow as tf

#	Import the keras from tensorflow

from tensorflow import keras

# 	The Fashion MNIST data is available directly Keras datasets API. We could load this datasets like this:

mnist = tf.keras.datasets.fashion_mnist

# 	Print the version of TensorFlow

print(tf.__version__)


(Training_images, Training_labels), (Test_images, Test_labels) = mnist.load_data()

# 				Normalizing
#      z= ( x- min(x) )/( max(x)- min(x) ).==> 0<= z >=1.
# We'll notice that all of the values in the number are between 0 and 255. 
#If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1. 

# Training_images reshaping. 

Training_images=Training_images.reshape(60000, 28, 28, 1)

# Training_images normalizing. 

Training_images=Training_images / 255.0

# Test_images reshaping.

Test_images = Test_images.reshape(10000, 28, 28, 1)

# Test_images normalizing.

Test_images=Test_images/255.0

#											 Design the Model for CNN:
# Sequential: Sequestianl defines a SEQUENCE of layers in the Neural Network
# The number of convolutions we want to generate purely arbitrary, but it is good to start with something in the order of 32.
# Each layer of neurons need an activation function to tell them what to do. There's lots of options for activation funciton, but we are using Relu activation function.
# Relu effectively means  "If Z>0 return Z, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
# The size of the Convolution, in this case a 3x3 grid
# In the first layer, the shape of the input data.

 
# Convolution with a MaxPooling layer which is then designed to compress the image, while maintaining the content of the features that were highlighted by the convlution. 
# By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image.  
# the idea is that it creates a 2x2 array of pixels, and picks the biggest one, thus turning 4 pixels into 1. It repeats this across the image, and 
# in so doing halves the number of horizontal, and halves the number of vertical pixels, effectively reducing the image by 25%.

# Dense: Adds a layer of neurons.

# Softmax takes a set of values, and effectively picks the biggest one, so, for example, 
# if the output of the last layer looks like [0.01, 0.1, 0.15, 0.11, 0.95, 0.11, 0.25, 0.05, 0.15], 
# it saves us from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0]
# The goal is to save a lot of coding.


Model = tf.keras.models.Sequential([

# Add convolution layer 

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  
# Add Maxpolling layer which is then designed to compress the images.

  tf.keras.layers.MaxPooling2D(2, 2),
  
# Add another convolution layer 

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  
# Add another Maxpolling layer and we can add arbitary oredr of Convolution  and MaxPooling layers.

  tf.keras.layers.MaxPooling2D(2,2),
  
# Flatten: Our images are a square. Flatten just takes that square and turns it into a one dimensional set.
  
  tf.keras.layers.Flatten(),
  
# Fully connected layer used features for classifying input image

  tf.keras.layers.Dense(128, activation='relu'),
  
# Express output as probability of image of belonging to particular class
 
  tf.keras.layers.Dense(10, activation='softmax')
])

# The model is defined above, and next thing is to do actually build it. 
# We do this by compiling it with an optimizer and loss function 

Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the built model

Model.summary()

#   Train the model 
# We have data that looks like the training data, then it can make a prediction for what that data would look like.

Model.fit(Training_images, Training_labels, epochs=8)

# After training the model, we should see an accuracy value at the end of the final epoch.
# But how would the model work with unseen data? That's why we have the test images. 
# We can call model.evaluate, and pass in the two sets, and it will report back the loss for each. 

Test_loss, Test_acc = Model.evaluate(Test_images, Test_labels)

# print the test_acc

print(Test_acc)
