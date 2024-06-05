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

#Główna funkcja analizująca każde zdjęcie:
def extract_features(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 40, 80], dtype='uint8')
    upper_hsv = np.array([100, 255, 255], dtype='uint8')

    skin_region_hsv = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    blurred_image = cv2.blur(skin_region_hsv, (2, 2))

    _, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY)

    hand_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hand_contours = max(hand_contours, key=lambda value: cv2.contourArea(value))

    hull = cv2.convexHull(hand_contours, returnPoints=False)

    defects = cv2.convexityDefects(hand_contours, hull)

    fingers_count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            start_index, end_index, far_index, _ = defects[i][0]
            start = tuple(hand_contours[start_index][0])
            end = tuple(hand_contours[end_index][0])
            far = tuple(hand_contours[far_index][0])
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            if angle <= np.pi / 2:
                fingers_count += 1

    if fingers_count > 0:
        fingers_count += 1

    return cv2.contourArea(hand_contours), cv2.arcLength(hand_contours, True), fingers_count

#Dwie funkcje służące do sortowania nazw plików alfabetycznie:
def atoi(character):
    return int(character) if character.isdigit() else character

def classic_sort(text):
    return [atoi(character) for character in re.split(r'(\d+)', text)]

#Posortowanie nazw zdjec z folderu alfabetycznie:
filenames = [filename for filename in os.listdir('Images') if filename.endswith('.png')]
filenames.sort(key=classic_sort)

#Analiza wszystkich zdjec przez algorytm:
print('Analizuje wszystkie zdjęcia...')
extracted_features = []
for filename in filenames:
    if filename.endswith('.png'):
        image = cv2.imread(f'Images\\{filename}')
        if image is None:
            print(f'Failed opening image {filename}')
        features_from_single_image = extract_features(image)
        extracted_features.append(features_from_single_image)

#Wczytanie zbioru etykiet (poprawnych odpowiedzi dla każdego zdjęcia):
labels = []
with open('labels.txt', 'r', encoding='UTF-8') as file:
    for line in file:
        labels.append(line.strip())


#Przygotowanie danych dla klasyfikatorow:
#"Konwersja" na data frame:
extracted_features_data_frame = pd.DataFrame(extracted_features, columns=['hand_contour_area', 'hand_outline_length', 'number_of_detected_fingers'])

#Normalizacja:
scaler = MinMaxScaler()
scaler.fit(extracted_features_data_frame)
normalized_extracted_features_data_frame = pd.DataFrame(scaler.transform(extracted_features_data_frame),columns=extracted_features_data_frame.columns)

#Podział na zbiory: testowy/treningowy w proporcji: 70% - trening, 30% - test
X_train, X_test, y_train, y_test = train_test_split(normalized_extracted_features_data_frame, labels, test_size=0.3)

#Użycie klasyfikatorów (porównanie):
print('\n------------------------\nWyniki klasyfikatorów\n------------------------')

#KNN (dla trzech sąsiadów):
knn_classifier = KNeighborsClassifier(3)
knn_classifier.fit(X_train, y_train)
print(f'KNN accuracy: {knn_classifier.score(X_test, y_test) * 100}%')
#GaussianNB:
gaussian_nb_classifier = GaussianNB()
gaussian_nb_classifier.fit(X_train, y_train)
print(f'GaussianNB accuracy: {gaussian_nb_classifier.score(X_test, y_test) * 100}%')
