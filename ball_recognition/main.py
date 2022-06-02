import cv2
import numpy as np
from keras.models import load_model
import pathlib


SUBDIR = 'with_flip_augmentation'

bbox_model = load_model(f'{pathlib.Path().resolve()}/models/{SUBDIR}/bbox_model')
classification_model = load_model(f'{pathlib.Path().resolve()}/models/{SUBDIR}/classification_model')

TARGET_WIDTH = 200
TARGET_HEIGHT = 150

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise RuntimeError('Camera doesnt work')

cv2.namedWindow('capture')

while True:
    key = cv2.waitKey(1)
    _, current_image = cap.read()
    print(current_image.shape)
    current_image = cv2.flip(current_image, 1)

    if key == ord('q'):
        break

    resized_image = cv2.resize(current_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    resized_image = np.array([resized_image])
    # print(resized_image.shape)
    class_prediction = classification_model.predict(resized_image)[0]
    print(f'probability there\'s a ball on picture: {class_prediction}')

    if class_prediction > 0.7:
        xmin, ymin, xmax, ymax = bbox_model.predict(resized_image)[0]

        xmin *= current_image.shape[1]
        xmax *= current_image.shape[1]
        ymin *= current_image.shape[0]
        ymax *= current_image.shape[0]
        print(xmin, ymin, xmax, ymax)

        cv2.rectangle(current_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
    
    cv2.imshow('capture', current_image)

cap.release()
cv2.destroyAllWindows()