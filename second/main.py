from collections import defaultdict
from pyexpat import features
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.measure import label, regionprops


train_dir = Path('./out') / 'train'
train_data = defaultdict(list)

def extract_features(binary):
    features = []

    _, hierarchy, = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ext_cnt = 0
    int_cnt = 0
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][-1] == -1:
            ext_cnt += 1
        elif hierarchy[0][i][-1] == 0:
            int_cnt += 1
    features.extend([ext_cnt, int_cnt])

    regions = regionprops(label(binary))
    features.append(regions[0].extent)

    centroid_normalized = np.array(regions[0].local_centroid) / np.array(regions[0].image.shape) 
    features.extend(centroid_normalized)
    features.append(regions[0].eccentricity)
    features.append(regions[0].orientation)
    return features


def glue_letters(labeled):
    regions = regionprops(labeled)
    regions_bboxes = [None] * len(regions)
    indexes_to_exclude = []
    for index1, region1 in enumerate(regions):

        top_left_y1, top_left_x1, bottom_right_y1, bottom_right_x1 = region1.bbox
        bbox_center1 = (top_left_y1 + (bottom_right_y1 - top_left_y1) / 2, top_left_x1 + (bottom_right_x1 - top_left_x1) / 2)

        for index2, region2 in enumerate(regions):

            found_clashing = False
            top_left_y2, top_left_x2, bottom_right_y2, bottom_right_x2 = region2.bbox
            bbox_center2 = (top_left_y2 + (bottom_right_y2 - top_left_y2) / 2, top_left_x2 + (bottom_right_x2 - top_left_x2) / 2)

            if top_left_y1 > bottom_right_y2 and abs(bbox_center1[1] - bbox_center2[1]) < 10 and bbox_center1[0] > bbox_center2[0]:
                min_x = top_left_x1 if top_left_x1 < top_left_x2 else top_left_x2
                max_x = bottom_right_x1 if bottom_right_x1 > bottom_right_x2 else bottom_right_x2
                regions_bboxes[index1] = (top_left_y2, min_x, bottom_right_y1, max_x)

                indexes_to_exclude.append(index2)
                found_clashing = True
                break

        if not found_clashing:
            regions_bboxes[index1] = region1.bbox

    indexes_to_exclude = list(sorted(indexes_to_exclude))
    for excluding_index in reversed(indexes_to_exclude):
        regions_bboxes[excluding_index] = None

    return list(filter(lambda x: x is not None, regions_bboxes))


def split_to_words(labeled):
    words = []
    region_bboxes = glue_letters(labeled)
    region_bboxes = list(reversed(sorted(region_bboxes, key=lambda bbox: labeled.shape[1] - bbox[3])))
    distances = []
    for i in range(0, len(region_bboxes) - 1, 1):
        distances.append(region_bboxes[i + 1][1] - region_bboxes[i][3])

    threshold = np.std(distances) * 0.5 + np.mean(distances)
    current_word = []

    for i in range(len(distances)):
        if i == 0:
            current_word.append(region_bboxes[0])
        if distances[i] > threshold:
            words.append(current_word)
            current_word = []

        current_word.append(region_bboxes[i + 1])

        if i == len(distances) - 1:
            words.append(current_word)

    return words


def image_to_text(image, knn):
    image[image > 0] = 1
    labeled = label(image)
    words = split_to_words(labeled)
    text = ''
    for word in words:
        for letter_bbox in word:
            img = labeled[letter_bbox[0]:letter_bbox[2], letter_bbox[1]:letter_bbox[3]]
            features = np.array(extract_features(img.astype("uint8")), dtype='f4').reshape(-1, 7)
            ret, _, _, _ = knn.findNearest(features, 2)
            text += chr(int(ret))
        text += ' '
    return text.strip()


for path in sorted(train_dir.glob('*')):
    if path.is_dir():
        for img_path in path.glob('*'):
            symbol = path.name[-1]
            gray = cv2.imread(str(img_path), 0)
            binary = gray.copy()
            binary[binary > 0] = 1
            train_data[symbol].append(binary)

features_array = []
responses = []
for i, symbol in enumerate(train_data):
    for img in train_data[symbol]:
        features = extract_features(img)
        features_array.append(features)
        responses.append(ord(symbol))

features_array = np.array(features_array, dtype='f4')
responses =  np.array(responses)

knn = cv2.ml.KNearest_create()
knn.train(features_array, cv2.ml.ROW_SAMPLE, responses)

expected_vals = ['C is LOW-LEVEL', 'C++ is POWERFUL', 'Python is INTUITIVE', 'Rust is SAFE', 'LUA is EASY', 'Javascript is UGLY']

for i in range(6):
    text = image_to_text(cv2.imread(f'./out/{i}.png', 0), knn)
    min_len = min(len(text), len(expected_vals[i]))
    mistakes = 0
    for j in range(min_len):
        if expected_vals[i][j] != text[j]:
            mistakes += 1
    if len(text) != len(expected_vals[i]):
        mistakes += abs(len(text) - len(expected_vals[i]))
    print(f'Expected: {expected_vals[i]}, received: {text}, error rate: {round(mistakes / len(expected_vals[i]), 2)}')