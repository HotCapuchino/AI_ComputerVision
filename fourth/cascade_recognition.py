import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops


def calculate_glass_mask(image, face_classifier, glass_classifier, scale_factor=None, min_neighbours=None):
    result = image.copy()
    face_region = face_classifier.detectMultiScale(result, scale_factor, min_neighbours)
    face_region = face_region[0] if len(face_region) > 0 else None

    if face_region is None:
        return None
    
    if len(face_region) == 0:
        return result

    eye_regions = glass_classifier.detectMultiScale(result, scale_factor, min_neighbours)
    min_dist = None
    min_indices = ()

    for index1, (x, y, w, h) in enumerate(eye_regions):
        if x > face_region[0] and y > face_region[1] and x + w < face_region[0] + face_region[2] and y + h < face_region[1] + face_region[3]:
            for index2, eye_region in enumerate(eye_regions):
                centroid1 = (x + w / 2, y + h / 2)
                centroid2 = (eye_region[0] + eye_region[2] / 2, eye_region[1] + eye_region[3] / 2)

                if centroid1 < centroid2:
                    if min_dist is None or min_dist < ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) ** 0.5:
                        min_dist = ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) ** 0.5
                        min_indices = (index1, index2)

    glass_region = None

    if len(min_indices) == 2:
        region1 = eye_regions[min_indices[0]]
        region2 = eye_regions[min_indices[1]]
        glass_region = (region1[0], min(region1[1], region2[1]), region2[0] + region2[2], max(region1[1] + region1[3], region2[1] + region2[3]))
        cv2.rectangle(result, (glass_region[0], glass_region[1]), (glass_region[2], glass_region[3]), (255, 255, 255))
        
        return glass_region
    
    return None


def prepare_glasses_image(glasses, width, height):
    mask = glasses.copy()[:, :, 0]
    mask[mask > 0] = 1
    mask = cv2.bitwise_not(mask)
    mask = label(mask)

    bbox = regionprops(mask)[1].bbox
    glasses_cut = glasses[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    glasses_cut = cv2.resize(glasses_cut, (width, height), interpolation = cv2.INTER_AREA)

    return glasses_cut


def overlay_pictures(image, glasses):
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2RGBA)
    glass_mask = calculate_glass_mask(img, face_classifier, eye_classifier, 1.2, 5)
    if glass_mask is None:
        return img

    cut_glasses = prepare_glasses_image(glasses, glass_mask[2] - glass_mask[0], glass_mask[3] - glass_mask[1])
    cut_glasses = cv2.bitwise_not(cut_glasses)

    image_mask = np.zeros_like(img)
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_RGB2RGBA)
    image_mask[glass_mask[1]:glass_mask[3], glass_mask[0]:glass_mask[2]] = cut_glasses

    img = cv2.add(img, image_mask)
    return img

glasses = cv2.imread('fourth/res/glasses.png', cv2.IMREAD_UNCHANGED)

face_cascade = 'fourth/object_detection/haarcascades/haarcascade_frontalface_default.xml'
eye_cascade = 'fourth/object_detection/haarcascades/haarcascade_eye.xml'

face_classifier = cv2.CascadeClassifier(face_cascade)
eye_classifier = cv2.CascadeClassifier(eye_cascade)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError('Camera doesnt work')

cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)

while True:
    _, image = cam.read()
    image = cv2.flip(image, 1)

    result = overlay_pictures(image, glasses)

    cv2.imshow('camera', result)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()