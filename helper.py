import os, cv2
import numpy as np
from skimage import io
from keras import backend as K

# get images and labels from the specified path
# two similar functions for extracting train and test images
def get_train_images_labels(path_name):
	# TODO write the function in a fashion that allows to extract images from any given dataset

	train_image_data = []
	train_image_labels = []

	# obtain images from the YALEFACES dataset
	if path_name == "yalefaces":
		train_images_path = [ os.path.join(path_name, item) for item in os.listdir(path_name) if not item.endswith('.sad') or item.endswith('.wink') ]
	

		# grayscaling and extracting labels for train images
		for i in train_images_path:
			img_read = io.imread(i, as_gray=True)
			train_image_data.append(img_read)

			label_read = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
			train_image_labels.append(label_read)

	# obtain images from other datasets
	else:
		image_paths = [ os.path.join(path_name, item) \
						for item in os.listdir(path_name) ]	 
		
		for i in image_paths:
			img_read = io.imread(i, as_gray = True)
			train_image_data.append(img_read)

			label_read = int(os.path.split(i)[1].split("_")[0].replace("image", "")) - 1
			train_image_labels.append(label_read) 

	return train_image_data, train_image_labels

# specific for yalefaces. other datasets are split randomly in main.py
def get_test_images_labels(path_name):
	test_images_path = [ os.path.join(path_name, item) \
			for item in os.listdir(path_name)  
			if item.endswith('.sad') or item.endswith('.wink') ]	
	
	test_image_data = []
	test_image_labels = []

	for i in test_images_path:
		img_read = io.imread(i, as_gray=True)
		test_image_data.append(img_read)

		label_read = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
		test_image_labels.append(label_read)

	return test_image_data, test_image_labels


# resize images to dim1xdim2
def resize_images(images, dim1, dim2):
	resized_images = []
	
	for im in images:
		if K.image_data_format() == "channels first":
			resized_im = np.reshape(cv2.resize(im, (dim1, dim2)), (1, dim1, dim2))
			resized_images.append(resized_im)
		else:
			resized_im = np.reshape(cv2.resize(im, (dim1, dim2)), (dim1, dim2, 1))
			resized_images.append(resized_im)

	return np.array(resized_images)

# normalize image pixels to be in range [-1, 1]
def normalize_images(images):
	normalized_images = []
	
	for im in images:
		# + 0.001 is to avoid division by 0
		im = (im - np.mean(images, axis=0)) / (np.std(images, axis=0) + 0.001)
		normalized_images.append(im)

	return np.array(normalized_images)

# NOTE not used
# normalize image pixels to be in range [0, 1]
def normalize_images2(images):
	return np.array(images / 255.0)

# NOTE not used
# flip images horizontally
def get_mirror_images(images):
	# for a 3D array (array of 2D images)
	return np.flip(images, 2)

# NOTE not used
# flip images horizontally, save to a directory and label
def get_mirror_images_from_path(path_name):
	# get paths to images
	image_paths = [ os.path.join(path_name, item) for item in os.listdir(path_name) ]
	
	for i in image_paths:
		# read image from path as array
		img_read = io.imread(i, as_grey=True)
		# read label as string
		# NOTE in progress

		new_path = 'mirrorimages'
		directory = os.path.dirname(new_path)
		if not os.path.exists(new_path):
			os.makedirs(new_path)

		label_read = new_path + os.path.split(i)[1].split(".")[0].replace("subject", "subject") + "." + os.path.split(i)[1].split(".")[1]
		#label_read = int(os.path.split(i)[1].split(".")[0].replace("subject", ""))
		#print label_read
		io.imsave(label_read + ".gif", np.flip(img_read, 1))
			
	return

# crop faces from one image
# takes pathname of an image, 
# returns cropped faces as a list of lists (matrices?)
def facecrop(image):
    # classifier used to detect faces
    # put in the working directory
    facedata = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(facedata)
    # read image from its path, save as a matrix
    img = cv2.imread(image)
    # list of detected faces
    faces = cascade.detectMultiScale(img)

    for f in faces:
		# coordinates and scale of faces
        x, y, w, h = f
        # crop faces
        sub_face = img[y:y+h, x:x+w]

        # split the name and extension of the image
        fname, ext = os.path.splitext(os.path.split(image)[1])
      
        # create a new directory to put cropped images if no directory exists 
        new_path = 'faces_cropped'
        directory = os.path.dirname(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        # in case more than one face is detected
        # str(i) is to distinguish filenames of faces detected from the same image
        cv2.imwrite(new_path + "/" + fname + str(x) + ext, sub_face)

