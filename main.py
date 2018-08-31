import cv2, os
import helper
from cnn import CNN
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
ap.add_argument("-d", "--directory", type=str, \
	help="path to the dataset")
ap.add_argument("-c", "--classes", type=int, \
	help="number of classes in the dataset")
ap.add_argument("-e", "--epochs", type=int, default=20, \
	help="number of epochs")
args = vars(ap.parse_args())

path_name = args["directory"]
num_classes = args["classes"]
num_epochs = args["epochs"]

# split test and train data 
if path_name == "yalefaces":
	train_image_data, train_image_labels = helper.get_train_images_labels(path_name)
	test_image_data, test_image_labels = helper.get_test_images_labels(path_name)
else:
	path_name_c = path_name 
	image_data, image_labels = helper.get_train_images_labels(path_name_c)
	(train_image_data, test_image_data, train_image_labels, test_image_labels) = train_test_split(image_data, image_labels, train_size=0.7)

# resizing to 64x64x1 and normalizing to be in range [-1, 1]
train_image_data = helper.normalize_images(helper.resize_images(train_image_data, 64, 64))
train_image_labels = np.array(train_image_labels)

test_image_data = helper.normalize_images(helper.resize_images(test_image_data, 64, 64))
test_image_labels = np.array(test_image_labels)

# turning labels vectors into matrices of shape (num_images, num_classes)
train_labels_matrix = np_utils.to_categorical(train_image_labels, num_classes)
test_labels_matrix = np_utils.to_categorical(test_image_labels, num_classes)

# building the model
# stochastic gradient descent
sgd = optimizers.SGD(lr = 0.1)

# CNN takes images of size 64x64
dim = 64
# numChannels = 1 because the images are converted to grayscale
model = CNN.build(numChannels = 1, imgRows = dim, imgCols = dim, numClasses = num_classes, weightsPath=args["weights"] if args["load_model"] > 0 else None)
# loss function in the paper was "mean_squared_error", but it didn't work well
# so i used "categorical_crossentropy"
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.summary()

# learning rate configurations from the paper
def learning_rate_decay(epoch):
    init_learning_rate = 0.1
    c1 = 50
    c2 = 0.65
	# learning rate formula 2
    #learning_rate = init_learning_rate / (epoch/(num_epochs/2) + c1/(max(1, (c1- (max(0, c1**2 * (epoch - c2*num_epochs)))/((1-c2)*num_epochs)))))
	
	# learning rate changes more with this formula
	# learning rate formula 1
    learning_rate = init_learning_rate / (epoch/(num_epochs/2) + c1/(max(1, (c1- (max(0, c1 * (epoch - c2*num_epochs)))/((1-c2)*num_epochs)))))

    return learning_rate

# LearningRateScheduler changes learning rate between epochs
learning_rate = LearningRateScheduler(learning_rate_decay, verbose=1)
callbacks_list = [learning_rate]

# train the model
print("Training: ")
if args["load_model"] < 0:
	model.fit(train_image_data, train_labels_matrix, batch_size=20, epochs=num_epochs, callbacks = callbacks_list, verbose=2)

if args["save_model"] > 0:
	model.save_weights(args["weights"], overwrite=True)

# test the model
print("Evaluating: ")
(loss, accuracy) = model.evaluate(test_image_data, test_labels_matrix, batch_size=20, verbose=1)
print("accuracy: {:.2f}%".format(accuracy * 100))
