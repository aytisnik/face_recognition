# Face Recognition in Python using Convolutional Neural Network and Simple Logistic Classifier 

This project aims to implement the face recognition system presented in the paper *Face Recognition using Convolutional Neural Network and Simple Logistic Classifier* (Hurieh Khalajzadeh et.al.).

## Getting Started

### Prerequisites

- Python 2.7
- MacOS or Linux (not tested on a Windows machine) 
- Keras
- OpenCV for Python

### Details

- **cnn.py** : implementation of the network model
- **helper.py** : implementation of helper functions used in main.py
- **cropfaces.py** : detection and cropping of faces from the images in the dataset
- **main.py** : building, training and testing the model 
- **weights** : directory of saved weights
- **caltechfaces** : directory of Caltech's Faces 1999 dataset (modified).
- **caltechfaces_cropped** : directory of cropped faces retrieved from Caltech's Faces 1999 dataset.
- **yalefaces** : directory of The Yale Face Dataset.

#### Network model
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 58, 58, 6)         300       
_________________________________________________________________
average_pooling2d_1 (Average (None, 29, 29, 6)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 22, 22, 16)        6160      
_________________________________________________________________
average_pooling2d_2 (Average (None, 11, 11, 16)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1936)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 26)                50362     
=================================================================
Total params: 56,822
Trainable params: 56,822
Non-trainable params: 0
_________________________________________________________________
```

### Instructions

- Labelling of the images in the dataset: 
  ```
  "image" + str(class_number) + "_" + str(image_number) + extension
  ```
  where `class number` is the label of the image, `image_number` is the unique index number of the image, it is optional.

  Example: `image16_02600.jpg`


- Run **`$ python cropfaces.py -d <directory_name>`** to detect and crop faces from the dataset if the faces in the dataset are not cropped, where `<directory_name>` is the pathname of the dataset.
  - This command puts cropped faces into a new directory named as `"<directory_name>" + "_cropped"`.
 
- Run **`$ python main.py -d <directory_name> -c <num_classes>`**, where `<directory_name>` is the pathname of the dataset, `<num_classes>` is the number of classes in this dataset. 
  - Optional arguments:
    * `-e` : number of epochs. (default=20)
    * `-s` : whether or not the model should be saved on the disk. (default=-1)
    * `-l` : whether or not pretrained model should be loaded. (default=-1)
    * `-w` : path to weights file.

  Example usage:
  ```
  $ python main.py -d caltechfaces_cropped -c 26 -s -1 -l -1 -w weights
  ```

## Results

The model has been trained on 2 datasets separately: _The Yale Face Dataset_ and Caltech's _Faces 1999_. 
- for _The Yale Face Dataset_, 2 pictures from each class (pictures with extension `.wink` and `.sad`) were used as test data, the rest (9 images for each class) were used as training data. 
  - In 10 epochs, both train and test accuracies reach 100%.

- for _Faces 1999_, training and test sets were split randomly. 
  Some results from the tests:
  - When test split is 5%, in 20 epochs both train and test accuracies reach 100%.  
  - When test split is 10%, in 20 epochs train accuracy is 100% and test accuracy is 93.33%.
  - When test split is 30%, in 20 epochs train accuracy is 100% and test accuracy is 97.01%.

- **Important note**: _Faces 1999_ dataset has been slightly modified such that classes with only one image have been eliminated, and all images have been renamed to include class labels for the ease of parsing. In addition, HaarCascadeClassifier has been utilized to detect and crop faces from the images. However, the classifier categorized some non-face objects as faces which later have been manually removed from the input data.

## Authors

* **Zumrud Shukurlu** 

## Helpful Resources

* Rosebrock A., LeNet - Convolutional Neural Network in Python, (July 24, 2018). Retrieved from <https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/>
* <https://codereview.stackexchange.com/questions/156736/cropping-faces-from-images-in-a-directory>

## References 

* Khalajzadeh H., Mansouri M., Teshnehlab M. (2014) Face Recognition Using Convolutional Neural Network and Simple Logistic Classifier. In: Snášel V., Krömer P., Köppen M., Schaefer G. (eds) Soft Computing in Industrial Applications. Advances in Intelligent Systems and Computing, vol 223. Springer, Cham 
* Weber, M. (1999). Faces 1999. Retrieved from <http://www.vision.caltech.edu/html-files/archive.html>
* The Yale Face Database. Retrieved from <http://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html>
