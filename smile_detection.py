import cv2 as cv
import dlib

from imutils import face_utils
from numpy.linalg import norm
from time import time


def main():
    # Initializing Dlibs face detector and creating the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Variables for saving the images when a person on the video is smiling
    start = time()
    img_counter = 0  # Counter for the image names

    cap = cv.VideoCapture(0)

    while True:
        ret, image = cap.read()
        # Converting to gray, because the detector works with gray images.
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detecting faces in the grayscale image
        faces = detector(gray_image, 0)

        for face in faces:
            # Determining the facial landmarks for the face region and converting
            # the facial landmark (x, y)-coordinates to a nd.array
            x_face, y_face = face.left(), face.top()
            w_face, h_face = face.right() - x_face, face.bottom() - y_face
            cv.rectangle(image, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)

            landmarks = predictor(gray_image, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Drawing the landmarks on the image for (x, y) in landmarks:
            #  cv.circle(image, (x, y), 2, (0, 255, 0), -1)

            # For calculating of the lip-jaw ratio
            jaw_width = norm(landmarks[2] - landmarks[14])
            lips_width = norm(landmarks[54] - landmarks[48])

            lip_jaw_ratio = lips_width / jaw_width

            # When a person smiles the mouth opening is bigger than the distance between the nose and the mouth
            mouth_opening = norm(landmarks[57] - landmarks[51])
            mouth_nose = norm(landmarks[33] - landmarks[51])

            # For calculating of the mouth-to-cheeks and mouth-to-jaw ratio
            mouth_to_cheeks = norm(landmarks[48] - landmarks[3]) + norm(landmarks[54] - landmarks[13])
            mouth_to_jaw = norm(landmarks[48] - landmarks[5]) + norm(landmarks[54] - landmarks[11])

            if lip_jaw_ratio > 0.43:
                if mouth_opening / mouth_nose >= 1:
                    # if mouth_to_cheeks < mouth_to_jaw * 1.6:
                    cv.putText(image, 'Smiling', (x_face, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # Saving smiling person images every 3 seconds
                    if time() - start > 3:
                        cv.imwrite(f'smiling_images\smile{img_counter}.jpg', image)
                        start = time()
                        img_counter += 1

        cv.imshow("Output", image)
        if cv.waitKey(1) == ord('q'):
            break

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
