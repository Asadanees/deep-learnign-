#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  mnist_1.py
# Important note: this code is tested on  Google Colab. Might be you need littel modefication to compplie and run on your loacl machine.
#  Copyright 2020 Asad Anees <asad@asad-Latitude-E7440>
#  
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
#  
#      #  A Computer Vision Project of recognize different items of clothing, trained from a dataset containing 10 different types (MNIST)

# We can recognize different items of clothing, trained from a dataset containing 10 different types.
# We can see  how to do fashion recognition using a Deep Neural Network (DNN) containing three layers from this project
# the input layer (in the shape of the data), the output layer (in the shape of the desired output) and a hidden layer. 
# We could  experiment with the impact of different sized of hidden layer, number of training epochs etc on the final accuracy.
# The test accuracy that is printed out at the end. 

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

# Training_images normalizing. 

Training_images=Training_images / 255.0

# Test_images normalizing.

Test_images=Test_images / 255.0

#											 Design the Model:


# Sequential: Sequestianl defines a SEQUENCE of layers in the Neural Network
# Flatten: Our images are a square. Flatten just takes that square and turns it into a one dimensional set.
# Dense: Adds a layer of neurons.
# Each layer of neurons need an activation function to tell them what to do. There's lots of options for activation funciton, but we are using Relu now.
# Relu effectively means  "If Z>0 return Z, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
# Softmax takes a set of values, and effectively picks the biggest one, so, for example, 
# if the output of the last layer looks like [0.01, 0.1, 0.15, 0.11, 0.95, 0.11, 0.25, 0.05, 0.15], 
# it saves us from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0]
# The goal is to save a lot of coding.

Model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# The model is defined above, and next thing is to do actually build it. 
# We do this by compiling it with an optimizer and loss function 

Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#   Train the model 
# We have data that looks like the training data, then it can make a prediction for what that data would look like.

Model.fit(Training_images, Training_labels, epochs=5)

# After training the model, we should see an accuracy value at the end of the final epoch.
# But how would the model work with unseen data? That's why we have the test images. 
# We can call model.evaluate, and pass in the two sets, and it will report back the loss for each. Let's give it a try:

test_loss = Model.evaluate(Test_images, Test_labels)

# Now we creates a set of classifications for each of the test images.

Classifications = Model.predict(Test_images)

# #  Now we print the first entry in the classifications. The output, after our run it is a list of numbers. Why do we think this is, and what do those numbers represent?

print(Classifications[0])

print(Test_labels[0])

