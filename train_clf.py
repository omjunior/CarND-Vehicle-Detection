import glob
import time
import pickle
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib

from feature import *


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

# Reduce the sample size
# sample_size = 1000
# shuffle(cars)
# shuffle(notcars)
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]

# TODO: Tweak these parameters and see how the results change.
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

print("Extracting features...")
t = time.time()
car_features = []
for car in cars:
    img = cv2.imread(car)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature = extract_features(img, cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                               pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                               spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    car_features.append(feature)

notcar_features = []
for not_car in notcars:
    img = cv2.imread(not_car)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature = extract_features(img, cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                               pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                               spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
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
# TODO: Take into consideration the time correlation between the pictures
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a SVC
parameters = {'C': (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1)}
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
