import cv2
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB


#Główna funkcja analizująca każde zdjęcie
def extract_features(image):
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')

    skinRegionHSV = cv2.inRange(hsvim, lower, upper)

    blurred = cv2.blur(skinRegionHSV, (2,2))

    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = max(contours, key=lambda x: cv2.contourArea(x))

    hull = cv2.convexHull(contours, returnPoints=False)

    defects = cv2.convexityDefects(contours, hull)

    fingers_count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            start_index, end_index, far_index, _ = defects[i][0]
            start = tuple(contours[start_index][0])
            end = tuple(contours[end_index][0])
            far = tuple(contours[far_index][0])
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #Tw. cosinusów
            # Jeśli kąt <= 90st. to mamy palec
            if angle <= np.pi / 2:
                fingers_count += 1

    algorithm_answers.append(fingers_count)
    return cv2.contourArea(contours), cv2.arcLength(contours, True), fingers_count


def atoi(character):
    return int(character) if character.isdigit() else character


def classic_sort(text):
    return [atoi(character) for character in re.split(r'(\d+)', text)]


filenames = [filename for filename in os.listdir('Images') if filename.endswith('.png')]
filenames.sort(key=classic_sort)

extracted_features = []
algorithm_answers = []
iterator = 1
for filename in filenames:
    if filename.endswith('.png'):
        image = cv2.imread(f'Images\\{filename}')
        if image is None:
            print(f'Failed opening image {filename}')
        print(f'Analyzing hand{iterator}.png')
        features = extract_features(image)
        extracted_features.append(features)
        iterator+=1

labels = []
with open('labels.txt', 'r', encoding='UTF-8') as file:
    for line in file:
        labels.append(line.strip())

# i = 1
# for feature in extracted_features:
#     print(f'{i}. {feature}  {labels[i - 1]}')
#     i+=1


#Przygotowanie danych dla klasyfikatorów:
extracted_features_data_frame = pd.DataFrame(extracted_features, columns=['Feature1', 'Feature2', 'Feature3'])

scaler = MinMaxScaler()
scaler.fit(extracted_features_data_frame)
normalized_extracted_features_data_frame = pd.DataFrame(scaler.transform(extracted_features_data_frame),columns=extracted_features_data_frame.columns)

#Podział na zbiory: testowy/treningowy w proporcji: 70% - trening, 30% - test
X_train, X_test, y_train, y_test = train_test_split(normalized_extracted_features_data_frame, labels, test_size=0.3)

#Użycie klasyfikatorów:
#KNN:
knn_classifier = KNeighborsClassifier(3)
knn_classifier.fit(X_train, y_train)
print(f'KNN accuracy: {knn_classifier.score(X_test, y_test) * 100}%')

#GaussianNB:
gaussian_nb_classifier = GaussianNB()
gaussian_nb_classifier.fit(X_train, y_train) #na zbiorze treningowym
print(f'GaussianNB accuracy: {gaussian_nb_classifier.score(X_test, y_test) * 100}%')


# sum = 0
# for i in range(0, 748):
#     print(f'{algorithm_answers[i]} == {labels[i]}?')
#     if int(algorithm_answers[i]) == int(labels[i]):
#         sum += 1
# print(f'Wynik algorytmu: {sum}/748')


