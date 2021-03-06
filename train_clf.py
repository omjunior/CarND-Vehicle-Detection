import glob
import time
import pickle
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib

from settings import load_settings
from features import extract_features

"""
Load the training images, extracts features and train a classifier, saving the results into pickle files
"""

# Divide up into cars and notcars
cars = []
notcars = []
images = glob.glob('./train/vehicles/GTI_Far/*.png')
cars.extend(images)
images = glob.glob('./train/vehicles/GTI_Left/*.png')
cars.extend(images)
images = glob.glob('./train/vehicles/GTI_MiddleClose/*.png')
cars.extend(images)
images = glob.glob('./train/vehicles/GTI_Right/*.png')
cars.extend(images)
images = glob.glob('./train/vehicles/KITTI_extracted/*.png')
cars.extend(images)

images = glob.glob('./train/non-vehicles/GTI/*.png')
notcars.extend(images)
images = glob.glob('./train/non-vehicles/Extras/*.png')
notcars.extend(images)

print("Cars:", len(cars), ", Not Cars:", len(notcars))

settings = load_settings()

print("Extracting features...")
t = time.time()
car_features = []
for car in cars:
    img = cv2.imread(car)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature = extract_features(img, settings['color_space'], settings['spatial_size'], settings['hist_bins'],
                               settings['orient'], settings['pix_per_cell'], settings['cell_per_block'])
    car_features.append(feature)

notcar_features = []
for not_car in notcars:
    img = cv2.imread(not_car)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature = extract_features(img, settings['color_space'], settings['spatial_size'], settings['hist_bins'],
                               settings['orient'], settings['pix_per_cell'], settings['cell_per_block'])
    notcar_features.append(feature)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
pickle.dump(X_scaler, open("scaler.p", "wb"))
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
# TODO: Take into consideration the time correlation between the images
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a SVC
parameters = {'C': (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1)}
svc = LinearSVC()
clf = GridSearchCV(svc, parameters)
# Check the training time for the SVC
t = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
print("Parameters", clf.best_params_)
# Save to disk
joblib.dump(clf, "clf.p")
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
