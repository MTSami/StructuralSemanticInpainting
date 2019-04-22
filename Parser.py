# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def getShape(im):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape_predictor",
                    help="path to facial landmark predictor", default='shape_predictor_68_face_landmarks.dat')
    ap.add_argument("-i", "--image", help="path to input image", default='image2.jpg')
    args = vars(ap.parse_args())

    # the facial landmark predictor
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # Resize input image, convert from RGB to GBR and thn convert it to grayscale
    image = imutils.resize(im, width=500)
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Initialize shape, in case landmark detector fails
    shape = np.zeros([68, 2], dtype=int)

    #determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array

    # #Since no face detector was used, we have no face detection rectangle.
    # #We circumvent the promlem by using a generic one-size fit all dlib recangle name window
    window = dlib.rectangle(30, -30, 470, 470)
    shape = predictor(gray, window)
    shape = face_utils.shape_to_np(shape)

    #----------------------------------------------------------------------------------------------------

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(window)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(0 + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), 8)

    #
    # # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    #print(shape)
    return shape

# if __name__ == '__main__':
#     main()