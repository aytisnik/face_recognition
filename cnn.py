from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras import backend as K

class CNN:
	@staticmethod
	def build(numChannels, imgRows, imgCols, numClasses, activation="tanh", weightsPath=None):
		#initialize the model
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		if K.image_data_format() == "channels first":
			inputShape = (numChannels, imgRows, imgCols)

		# first conv layer CONV -> TANH
		model.add(Conv2D(6, kernel_size = (7, 7), activation = "tanh", padding = "valid", input_shape = inputShape))
		#model.add(Conv2D(6, kernel_size = (7, 7), activation = "tanh", padding = "valid"))

		# first subsampling POOL
		model.add(AveragePooling2D(pool_size = (2, 2), strides=2))

		# second conv layer CONV -> TANH
		model.add(Conv2D(16, kernel_size = (8, 8), activation = "tanh", padding = "valid"))

		# second subsampling POOL
		model.add(AveragePooling2D(pool_size = (2, 2), strides=2))
		
		model.add(Flatten())

		# output FC
		# activation was "tanh" in the paper, but "softmax" worked better
		# softmax is logistic regression for multiple classes
		model.add(Dense(numClasses, activation = "softmax"))

		if weightsPath is not None:
			model.load_weights(weightsPath)

		return model
