# -*- coding: utf-8 -*-

"""
University of Turku
Machine Learning and Pattern Recognition - Excercise Project

Created on Wed Mar 13 13:39:26 2019
@author: Mikael Kylänpää


--- Introduction ---

The tasks of this exercise are to acquire a set of images, extract features from
them, and train a set of classifiers to predict the contents of the images.


--- Data set ---

Using a premade set of image URLs located in text files:
    birdnests.txt: 39 birdnest images
    honeycomb.txt: 39 honeycomb images
    lighthouses.txt: 42 lighthouse images

There can be faulty URLs, so the image counts may vary. The images are colorful
and come in varying sizes.


--- Methods ---

Using a predefined set of methods:
    preparation: resize, grayscale
    feature extraction: RGB means & variances, grayscale level co-occurrence matrix, zscore
    visualization: principal component analysis, self-organizing map
    classifiers: k nearest neighbors, Ridge regression, multilayer perceptron
    evaluation: nested cross-validation, early stopping, accuracy-score, confusion matrix   



1. Data import

Download images and assign integers as class-labels:
    (birdnests: 0, honeycombs: 1, lighthouses: 2)  
"""

import numpy as np
from skimage import io


imgs = [] # raw images
labels = []
print("\nDownloading images")

# birdnests
for url in np.loadtxt('images/birdnests.txt', dtype='U100'):
    try:
        # add label after succesful entry
        imgs.append(io.imread(url))
        labels.append(0)
    except:
        # skip faulty URL
        continue
print("set downloaded")

# honeycombs
for url in np.loadtxt('images/honeycomb.txt', dtype='U100'):
    try:
        imgs.append(io.imread(url))
        labels.append(1)
    except:
        continue
print("set downloaded")

# lighthouses
for url in np.loadtxt('images/lighthouse.txt', dtype='U100'):
    try:
        imgs.append(io.imread(url))
        labels.append(2)
    except:
        continue
print("set downloaded")

# convert to numpy array for later usability
# TODO: find more efficient way
imgs = np.array(imgs)
labels = np.array(labels)


# Retrieve class indices:
# TODO: find more efficient way

inds1 = []
inds2 = []
inds3 = []

for i in range(len(labels)):
    if labels[i] == 0: # birdnests
        inds1.append(i)
    if labels[i] == 1: # honeycombs
        inds2.append(i)
    if labels[i] == 2: # lighthouses
        inds3.append(i)

# set seed for the rest of the exercise for reproduceable calculations
np.random.seed(0)



"""
2. Data preparation

In order to compare images pixel by pixel, they need to be in the same size. Find
the mode width and height of all images, and resize them correspondingly.

Save also grayscale versions of images. The quantization level must be reduced
to 8 bits for GLCM.
"""

import matplotlib.pyplot as plt
from statistics import mode
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte


# Calculate mode sizes:

ws = []
hs = []

for img in imgs:
    ws.append(len(img))
    hs.append(len(img[0]))


# Resize images:

for i in range(len(imgs)):
    imgs[i] = resize(imgs[i], (mode(ws), mode(hs)))

# verify changes
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(imgs[0])
ax[0,1].imshow(imgs[1])
ax[1,0].imshow(imgs[inds2[0]])
ax[1,1].imshow(imgs[inds3[0]])
plt.show()


# Convert to grayscale:

imgs_gr = []

for i in range(len(imgs)):
    imgs_gr.append(rgb2gray(imgs[i]))

# reduce quantization (8-bit u-int)
for i in range(len(imgs_gr)):
    imgs_gr[i] = img_as_ubyte(imgs_gr[i])

plt.imshow(imgs_gr[0], cmap='Greys_r')
plt.show()



"""
3. Feature extraction

From color images, extract the RGB means and variances per horizontal pixel (vertically).
This sums up to 500 (image height) x 3 (color channels) x 2 (mean & variance)
= 3000 features per image.

Extract grayscale level co-occurrence features from grayscale images. This is usually
done by selecting different patches from different parts of the image. In this
case, let's select 3 different sized patches from the center of the image, assuming
that the main element is in the center. A GLC-matrix is contructed from the patch,
in this case by 4 different angles. From every GLCM, 5 different values are extracted.
This makes 3 (patches) x 4 (angles) x 5 (values) = 60 features per grayscale image.

The two feature sets are first standardized individually by zscore. This scales
all feature columns into same limit for each set.
"""

from skimage.feature import greycomatrix, greycoprops
from scipy.stats import zscore


# Extract RGB features:

feats_rgb = [] # feature array
print("\nExtracting RGB")

for img in imgs:
    # r, g & b -matrices
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    # calculate means & variances per horizontal pixel vector
    r_mean = np.mean(r, axis=1)
    g_mean = np.mean(g, axis=1)
    b_mean = np.mean(b, axis=1)
    r_var = np.var(r, axis=1)
    g_var = np.var(g, axis=1)
    b_var = np.var(b, axis=1)
    
    # add values into feature vector
    feat_vect = []
    for i in range(len(r_mean)):
        feat_vect.append(r_mean[i])
        feat_vect.append(g_mean[i])
        feat_vect.append(b_mean[i])
        feat_vect.append(r_var[i])
        feat_vect.append(g_var[i])
        feat_vect.append(b_var[i])
    
    # add vector to feature array
    feats_rgb.append(feat_vect)

feats_rgb = np.array(feats_rgb)
print(feats_rgb.shape)


# Extract GLCM features:

feats_gr = []
print("\nExtracting GLCM")

for img in imgs_gr:
    # select patches from center
    patches = []
    patches.append(img[225:275, 225:275]) # 50x50
    patches.append(img[200:300, 200:300]) # 100x100
    patches.append(img[125:375, 125:375]) # 200x200
    
    feat_vect = []
    for patch in patches:
        # compute GLCM matrix
        glcm = greycomatrix(patch, distances=[5],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], # angle radians
                            levels=256, # 256 for an 8-bit image
                            symmetric=True, normed=True)
        # compute & add GLCM values
        feat_vect.extend(greycoprops(glcm, 'contrast')[0][:])
        feat_vect.extend(greycoprops(glcm, 'dissimilarity')[0][:])
        feat_vect.extend(greycoprops(glcm, 'homogeneity')[0][:])
        feat_vect.extend(greycoprops(glcm, 'ASM')[0][:])
        feat_vect.extend(greycoprops(glcm, 'correlation')[0][:])
    
    feats_gr.append(feat_vect)

feats_gr = np.array(feats_gr)
print(feats_gr.shape)


# Zscore standardization:
feats_rgb = zscore(feats_rgb, axis=0)
feats_gr = zscore(feats_gr, axis=0)



"""
4. Principal component analysis (PCA)

Reduce feature dimensionality to 2 principal components and plot the data. Plot
the feature sets first individually. Produce also combined PCA, by merging the prior
principal components, standardizing them to same scale with zscore and running
another PCA on them.
"""

from sklearn.decomposition import PCA


# PCA:

# separate
pca = PCA(n_components=2)
pca_rgb = pca.fit_transform(feats_rgb)
pca_gr = pca.fit_transform(feats_gr)

# combined
feats_pcas = np.column_stack([pca_rgb, pca_gr]) # merge results
feats_pcas = zscore(feats_pcas, axis=0) # standardize
pca_comb = pca.fit_transform(feats_pcas) # PCA again


# Visualize with scatterplots:

# RGB
plt.scatter(pca_rgb[inds1,0], pca_rgb[inds1,1], color='blue')
plt.scatter(pca_rgb[inds2,0], pca_rgb[inds2,1], color='red')
plt.scatter(pca_rgb[inds3,0], pca_rgb[inds3,1], color='green')
plt.title('PCA on RGB-features only')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['birdnests','honeycombs','lighthouses'])
plt.show()

# GLCM
plt.scatter(pca_gr[inds1,0], pca_gr[inds1,1], color='blue')
plt.scatter(pca_gr[inds2,0], pca_gr[inds2,1], color='red')
plt.scatter(pca_gr[inds3,0], pca_gr[inds3,1], color='green')
plt.title('PCA on GLCM-features only')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['birdnests','honeycombs','lighthouses'])
plt.show()

# combined
plt.scatter(pca_comb[inds1,0], pca_comb[inds1,1], color='blue')
plt.scatter(pca_comb[inds2,0], pca_comb[inds2,1], color='red')
plt.scatter(pca_comb[inds3,0], pca_comb[inds3,1], color='green')
plt.title('PCA on combined feature sets')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['birdnests','honeycombs','lighthouses'])
plt.show()



"""
5. Self-organizing maps (SOM)

Visualize features with self-organizing maps. Similar to PCA, SOM enables the strucure
of the data to be visualized in 2D. Let's visualize both the RGB and GLCM feature
sets individually.

Using MiniSom library (https://github.com/JustGlowing/minisom)
From the documentation:
    "if your dataset has 150 samples, 5*sqrt(150) = 61.23
    hence a map 8-by-8 should perform well."
Using 8x7 (=56) grid with this data. 

"""

from minisom import MiniSom


# Train RGB SOM:

print("\nTraining SOM")
som = MiniSom(8, 7, len(feats_rgb[0]))
som.train_random(feats_rgb,
                 20000, # iterations
                 verbose=True)

# draw nodes, background coloring
plt.figure(figsize=(8,7))
plt.pcolor(som.distance_map().T, cmap='Greys_r')
plt.colorbar()

# plot images
for i in range(len(feats_rgb)):
    if i in inds1: # birdnests
        colr = 'blue'
        pos = 0.3
    if i in inds2: # honeycombs
        colr = 'red'
        pos = 0.5
    if i in inds3: # lighthouses
        colr = 'green'
        pos = 0.7
    # draw dot into winner node/tile
    w = som.winner(feats_rgb[i])
    plt.plot(w[0]+pos, w[1]+.5, 'o', markerfacecolor=colr,
             markersize=15, markeredgewidth=0)
plt.title('RGB SOM, 20000 random samples')
plt.axis([0, 8, 0, 7])
plt.show()


# Train GLCM SOM:

print("\nTraining SOM")
som = MiniSom(8, 7, len(feats_gr[0]))
som.train_random(feats_gr, 50000, verbose=True)

plt.figure(figsize=(8,7))
plt.pcolor(som.distance_map().T, cmap='Greys_r')

for i in range(len(feats_gr)):
    if i in inds1:
        colr = 'blue'
        pos = 0.3
    if i in inds2:
        colr = 'red'
        pos = 0.5
    if i in inds3:
        colr = 'green'
        pos = 0.7
    w = som.winner(feats_gr[i])
    plt.plot(w[0]+pos, w[1]+.5, 'o', markerfacecolor=colr,
             markersize=15, markeredgewidth=0)

plt.title('GLCM SOM, 50000 random samples')
plt.axis([0, 8, 0, 7])
plt.show()



"""
6. K nearest neighbors classifier (kNN)

kNN doesn't function with high dimensional data. Therefore, reducing dimensions
with PCA. Reducing RGB-features to 10 PCs and GLCM-features to 5 PCs.

Using nested cross-validation to optimize k and validate performance:
    outer loop (validation): Stratified 10-fold CV. Generates 10 stratified folds
    from the data. Uses one fold at a time as the test set and the rest as the
    model selection set.
    inner loop (model selection): Iterates through the candidates for best k. Trains
    the classifier in leave-one-out CV (one input as the test set, the rest as
    the training set /iteration). Calculates predictions for current k.

For each outer loop fold-set, the best k is selected by accuracy-%. The best classifier
is then trained again to predict the validation set to simulate predicting unseen
data.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Reduce dimensionality with PCA:

pca = PCA(n_components=10) # 10 for RGB
pca_rgb = pca.fit_transform(feats_rgb)

pca = PCA(n_components=5) # 5 for GLCM
pca_gr = pca.fit_transform(feats_gr)

feats_knn = np.column_stack([pca_rgb, pca_gr]) # merge features
feats_knn = zscore(feats_knn, axis=0) # standardize again


# Nested CV:

ks = range(1, 26) # k candidates (1,2,...,25)
ks_best = [] # winner ks
acc_train = [] # winner k training accuracies
acc_valid = [] # winner k validation accuracies
inds_knn = [] # # stratification-ordered indices
labels_knn = [] # stratification-ordered labels
preds_knn = [] # final predictions
kf = StratifiedKFold(n_splits=10) # 10 folds
print("\nRunning kNN")

# outer loop
for inds_model, inds_valid in kf.split(feats_knn, labels):
    # assign model selection & validation sets
    X_model = feats_knn[inds_model]
    y_model = labels[inds_model]
    X_valid = feats_knn[inds_valid]
    y_valid = labels[inds_valid]
    # save index & label
    inds_knn.extend(inds_valid)
    labels_knn.extend(y_valid)
    
    acc_ks = [] # accuracies of k
    for k in ks:
        y_pred = [] # loo predictions
        
        # inner loop
        for i in range(len(X_model)):
            # assign test & train set
            X_test = [X_model[i]]
            X_train = np.delete(X_model, i, axis=0)
            y_train = np.delete(y_model, i)
            # train classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            # predict test & save result
            y_pred.append(knn.predict(X_test))
        
        # calculate & save training accuracy for k
        acc_ks.append(accuracy_score(y_model, y_pred))
    
    # find best k
    ind_best = np.argmax(acc_ks)
    k_best = ks[ind_best]
    ks_best.append(k_best)
    acc_train.append(acc_ks[ind_best]) # save training accuracy of best k
    # train whole fold-set with best k
    knn = KNeighborsClassifier(n_neighbors=k_best)
    knn.fit(X_model, y_model)
    # predict validation set
    preds = knn.predict(X_valid)
    preds_knn.extend(preds) # save predictions
    acc_valid.append(accuracy_score(y_valid, preds)) # save validation accuracy


# Evaluation:

cv_result = np.column_stack([ks_best, acc_train, acc_valid])
print("\nbest k,   acc-% (train),   acc-% (valid)")
print(cv_result)
print("\nOverall accuracy-% (validation): " + str(np.mean(acc_valid)))

# image index, prediction, true label
cv_preds = np.column_stack([inds_knn, preds_knn, labels_knn])

# confusion matrix
print(confusion_matrix(labels_knn, preds_knn))


# Examples:

# birdnests
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
fig.suptitle('kNN birdnests')
# correct
ax[0,0].imshow(imgs[3])
ax[0,1].imshow(imgs[4])
ax[0,2].imshow(imgs[6])
ax[0,3].imshow(imgs[7])
# wrong
ax[1,0].imshow(imgs[42])
ax[1,1].imshow(imgs[78])
ax[1,2].imshow(imgs[81])
ax[1,3].imshow(imgs[86])
[ax.set_axis_off() for ax in ax.ravel()] # (https://stackoverflow.com/a/52776192)
plt.show()

# honeycombs
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
fig.suptitle('kNN honeycombs')
ax[0,0].imshow(imgs[39])
ax[0,1].imshow(imgs[40])
ax[0,2].imshow(imgs[41])
ax[0,3].imshow(imgs[43])
ax[1,0].imshow(imgs[2])
ax[1,1].imshow(imgs[5])
ax[1,2].imshow(imgs[9])
ax[1,3].imshow(imgs[16])
[ax.set_axis_off() for ax in ax.ravel()]
plt.show()

# lighthouses
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
fig.suptitle('kNN lighthouses')
ax[0,0].imshow(imgs[77])
ax[0,1].imshow(imgs[79])
ax[0,2].imshow(imgs[80])
ax[0,3].imshow(imgs[82])
ax[1,0].imshow(imgs[47])
ax[1,1].imshow(imgs[28])
ax[1,2].imshow(imgs[35])
ax[1,3].axis('off') # no more left
[ax.set_axis_off() for ax in ax.ravel()]
plt.show()



#"""
#7. Ridge regression classifier
#
#Ridge regression (reguralized linear model) is similar to least squares, but it
#uses bias to reduce errors on multicollinear data. Ridge regression functions with
#high dimensional data, so using the full feature set.
#
#Using leave-one-out CV for optimizing the alpha. Alpha-hyperparameter affects the
#amount of bias.
#"""
#
#from sklearn.linear_model import RidgeClassifier
#
#
## Use all features with standardization:
#
#feats_all = np.column_stack([feats_rgb, feats_gr])
#feats_all = zscore(feats_all, axis=0)
#
#
## Use leave-one-out cross-validation to optimize alpha:
#
#alphas = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000, 5000]
#acc = []
#
#print("\nOptimizing Ridge classifier")
#
#for a in alphas:
#    y_pred = []
#    
#    for i in range(len(feats_all)):
#        X_test = [feats_all[i]]
#        X_train = np.delete(feats_all, i, axis=0)
#        y_train = np.delete(labels, i)
#        ridge = RidgeClassifier(alpha=a)
#        ridge.fit(X_train, y_train)
#        y_pred.append(ridge.predict(X_test)[0])
#    
#    acc.append(accuracy_score(labels, y_pred))
#
#cv_result = np.column_stack([alphas, acc])
#print(cv_result)
#a_best = 1000
#
#
#
#"""
#8. Multilayer perceptron classifier
#
#Feed-forward multilayer perceptron (MLP) with 1 hidden layer of 256 neurons
#(3060 => 256 => 3).
#
#Using cross-validation to
#
#"""
#
#import tensorflow as tf
#from sklearn.model_selection import train_test_split
#
#
## Craft features for MLP committee/ensemble:
#
## assign seeds
#np.random.seed(0)
#tf.set_random_seed(0)
#
## split input to train & validation sets (use all 3060 features)
## (~ 20 validation images)
#X_train, X_valid, y_train, y_valid = train_test_split(
#        feats_all, labels, test_size=0.2, stratify=labels)
#
## apply one hot encoding
#y_train_enc = tf.keras.utils.to_categorical(y_train, num_classes=3)
#y_valid_enc = tf.keras.utils.to_categorical(y_valid, num_classes=3)
#
## split train set to separate folds
#n_members = 3 # use 3 folds/members (~ 30 images/member)
#X_folds = np.array_split(X_train, n_members)
#y_folds = np.array_split(y_train, n_members)
#y_folds_enc = np.array_split(y_train_enc, n_members)
#
#
## Create MLP ensemble:
#
#dim_h = 256 # hidden layer neurons
#models = []
#models = list()
#
#for i in range(n_members):
#    # define structure
#    model = tf.keras.models.Sequential() # feed-forward
#    model.add(tf.keras.layers.Dense(dim_h, # hidden layer dim
#                                    input_dim=3060, # input dim
#                                    activation='relu'))
#    model.add(tf.keras.layers.Dense(3, # output dim
#                                    activation='softmax'))
#    # compile model
#    model.compile(loss='categorical_crossentropy',
#            optimizer='adam',
#            metrics=['accuracy'])
#    # save model
#    models.append(model)
#
#
## Train ensemble:
#
#batch_size = 32 # inputs per weight update
#n_epochs = 1000 # max whole set iterations
#
#print(tf.keras.__version__)
#
#print("\nOptimizing MLPs")
#
#for i in range(n_members):
#    print("\nMember " + str(i) + ":")
#    # fit model
#    models[i].fit(X_folds[i], y_folds_enc[i], batch_size, n_epochs, verbose=0,
#              # find early stopping point
#              callbacks=[tf.keras.callbacks.EarlyStopping(
#                      monitor='val_loss', # value to optimize
#                      patience=100, # checks after local minimum
#                      verbose=1,
#                      mode='min', # minimize/maximize
#                      restore_best_weights=True)], # apply minimum
#              validation_data=(X_valid, y_valid_enc))
#
### clear session/GPU memory
##tf.keras.backend.clear_session()
#
#
#
#"""
#9. Model evaluation/comparison
#
#"""
#
#import pandas as pd
#from sklearn.model_selection import StratifiedKFold
#
