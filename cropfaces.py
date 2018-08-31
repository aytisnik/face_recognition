import os, cv2
import argparse

# crop faces from all images in the specified directory
# path_name of the directory is given as a parameter
def facecrop(path_name):
    # list of paths to images 
    image_paths = [ os.path.join(path_name, item) \
                        for item in os.listdir(path_name) ]

    # path to new directory to put the cropped faces
    new_path = path_name + "_cropped"
    # if directory does not exist, make one
    directory = os.path.dirname(new_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # classifier used to detect faces
    # put in the working directory
    facedata = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(facedata)

    for im in image_paths:
        print im
        # split filename and extension of each image
        fname, ext = os.path.splitext(os.path.split(im)[1])

        # read image as a matrix
        img = cv2.imread(im)

        # detect faces
        faces = cascade.detectMultiScale(img)

        for i, face in enumerate(faces):
            x, y, w, h = face
            # crop faces
            sub_face = img[y:y+h, x:x+w]
            # write image to the new directory
            cv2.imwrite(os.path.join(new_path, fname + str(i) + ext), sub_face)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", type=str, \
    help="path to the dataset")
    args = vars(ap.parse_args())

    path_name = args["directory"]
    facecrop(path_name)	
