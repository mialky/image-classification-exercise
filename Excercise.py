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

Using the predefined set of methods:
    preparation: resize, grayscale
    feature extraction: RGB means & variances, grayscale level co-occurrence matrix, zscore
    visualization: principal component analysis, self-organizing map
    classifiers: k nearest neighbors, Ridge regression, multilayer perceptron (neural network)
    evaluation:  cross-validation, nested cross-validation, early stopping, accuracy-score, confusion matrix   


--- Data preparation ---

The images are downloaded as they are. No images are excluded from the set apart
from those that fail to load.



1. Data import

Download images and assign integers as class-labels:
    (birdnests: 0, honeycombs: 1, lighthouses: 2)  
"""

import numpy as np
from skimage import io

# set seed for the rest of the exercise for reproduceable calculations
np.random.seed(0)


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



"""
2. Data preparation

In order to compare images pixel by pixel, they need to be in the same size. Find
the mode width and height of all images, and resize them correspondingly.

Save also grayscale versions of images. The quantization level must be reduced
to 8 bits for GLCM.
"""

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte


# Calculate sizes:

ws = []
hs = []

for img in imgs:
    ws.append(len(img))
    hs.append(len(img[0]))


# Resize images to mode:

for i in range(len(imgs)):
    # (https://stackoverflow.com/a/6252400)
    imgs[i] = resize(imgs[i], (np.bincount(ws).argmax(), np.bincount(hs).argmax()))

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
    imgs_gr[i] = img_as_ubyte(imgs_gr[i]) # reduce quantization (8-bit u-int)

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
                 20000, # amount of iterations
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

import random
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
preds_knn = [] # final predictions
kf = StratifiedKFold(n_splits=10) # 10 folds

# stratification-ordered indices & labels
inds_kf = []
labels_kf = []

print("\nRunning kNN CV")

# outer loop
for inds_model, inds_valid in kf.split(feats_knn, labels):
    # assign model selection & validation sets
    X_model = feats_knn[inds_model]
    y_model = labels[inds_model]
    X_valid = feats_knn[inds_valid]
    y_valid = labels[inds_valid]
    # save index & label
    inds_kf.extend(inds_valid)
    labels_kf.extend(y_valid)
    
    acc_ks = [] # accuracies of different ks
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
    print("fold-set complete")


# Evaluation:

cv_result = np.column_stack([ks_best, acc_train, acc_valid])
print("\nbest k,   acc-% (train),   acc-% (valid)")
print(cv_result)
print("\nOverall accuracy-% (validation): " + str(accuracy_score(labels_kf, preds_knn)))

# confusion matrix
print(confusion_matrix(labels_kf, preds_knn))


# Examples:

# plots randomly 4 correct & 4 false classified images for desired class
def explot(inds, preds, labels, lab):
    corr = []
    false = []
    for i in range(len(labels)):
        if (labels[i] == lab and preds[i] == lab): # true positives
            corr.append(inds[i])
        elif (labels[i] != lab and preds[i] == lab): # false positives
            false.append(inds[i])
    # random select max 4 inds
    corr = random.sample(corr, min([len(corr), 4]))
    false = random.sample(false, min([len(false), 4]))
    # plot
    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    if lab == 0:
        fig.suptitle('classified as birdnests')
    elif lab == 1:
        fig.suptitle('classified as honeycombs')
    else:
        fig.suptitle('classified as lighthouses')
    # correct
    for i in range(4):
        if i in range(len(corr)):
            ax[0,i].imshow(imgs[corr[i]])
            ax[0,i].set_title(corr[i])
        else:
            ax[0,i].axis('off') # if under 4 corr found
    # false
    for i in range(4):
        if i in range(len(false)):
            ax[1,i].imshow(imgs[false[i]])
            ax[1,i].set_title(false[i])
        else:
            ax[0,i].axis('off') # if under 4 false found
    [ax.set_axis_off() for ax in ax.ravel()] # (https://stackoverflow.com/a/52776192)
    plt.show()

explot(inds_kf, preds_knn, labels_kf, 0) # birdnests
explot(inds_kf, preds_knn, labels_kf, 1) # honeycombs
explot(inds_kf, preds_knn, labels_kf, 2) # lighthouses



"""
7. Ridge regression classifier

Ridge regression (reguralized linear model) is similar to least squares, but it
uses bias to reduce errors on multicollinear data. Ridge regression functions with
high dimensional data, so using the full feature set.

Using the same nested CV than with kNN to optimize alpha and predict on validation
set. The stratification splits are also same.
"""

from sklearn.linear_model import RidgeClassifier


# Full feature set with standardization:

feats_all = np.column_stack([feats_rgb, feats_gr])
feats_all = zscore(feats_all, axis=0)


# Nested CV:

alphas = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000, 5000]
as_best = [] # winner alphas
acc_train = []
acc_valid = []
preds_ridge = []

# will generate same folds than kNN, will use same inds_kf & labels_kf
kf = StratifiedKFold(n_splits=10)

print("\nRunning Ridge CV")

for inds_model, inds_valid in kf.split(feats_all, labels):
    X_model = feats_all[inds_model]
    y_model = labels[inds_model]
    X_valid = feats_all[inds_valid]
    y_valid = labels[inds_valid]
    
    acc_as = [] # accuracies of different alphas
    for a in alphas:
        y_pred = [] 
        
        for i in range(len(X_model)):
            X_test = [X_model[i]]
            X_train = np.delete(X_model, i, axis=0)
            y_train = np.delete(y_model, i)
            # train classifier
            ridge = RidgeClassifier(alpha=a)
            ridge.fit(X_train, y_train)
            y_pred.append(ridge.predict(X_test))
        
        acc_as.append(accuracy_score(y_model, y_pred))
    
    # find best alpha
    ind_best = np.argmax(acc_as)
    a_best = alphas[ind_best]
    as_best.append(a_best)
    acc_train.append(acc_as[ind_best]) # save training accuracy of best alpha
    ridge = RidgeClassifier(alpha=a_best)
    ridge.fit(X_model, y_model)
    # predict validation set
    preds = ridge.predict(X_valid)
    preds_ridge.extend(preds)
    acc_valid.append(accuracy_score(y_valid, preds))
    print("fold-set complete")


# Evaluation:

cv_result = np.column_stack([as_best, acc_train, acc_valid])
print("\nbest alpha,   acc-% (train),   acc-% (valid)")
print(cv_result)
print("\nOverall accuracy-% (validation): " + str(accuracy_score(labels_kf, preds_ridge)))
print(confusion_matrix(labels_kf, preds_ridge))


# Examples:

explot(inds_kf, preds_ridge, labels_kf, 0)
explot(inds_kf, preds_ridge, labels_kf, 1)
explot(inds_kf, preds_ridge, labels_kf, 2)





"""
8. Multilayer perceptron committee/ensemble classifier

--- Model structure ---

Using TensorFlow and Keras to build multilayer perceptrons, which are simple artificial
neural networks (ANNs). Instead of a single network, using an ensemble (or committee).
This means training a set of multiple models and generalizing their outputs for
the final output.

In this case, using an ensemble of 5 members. The training inputs are split to 5
folds, and each model is fitted to one fold. A separate validation set is then
used to validate the generalization performance. The final prediction of the ensemble
is the maximum class among separate models' predictions.

Each member is a similar feed-forward multilayer perceptron (MLP) with 1 hidden
layer of 256 neurons (3060 inputs => 256 hidden layer neurons => 3 outputs).


--- Training & validation setup ---

Using stratified 10-fold CV in order to make validation predictions on the whole
input set. The stratification splits are again the same.

The models are re-trained multiple times, but each time the training is 'early
stopped' at the same state. (TODO: define)
Therefore, the performance on each validation fold should be similar.
"""

import tensorflow as tf


# set TensorFlow seed
tf.set_random_seed(0)


# 10-fold CV:

n_members = 5 # ensemble members
n_neurons = 256 # hidden layer neurons
n_epochs = 100 # training epochs/model
acc_valid = []
preds_mlp = []
kf = StratifiedKFold(n_splits=10)

print("\nRunning MLP")

for inds_train, inds_valid in kf.split(feats_all, labels):
    # using full feature set, split data to model & validation sets
    X_train = feats_all[inds_model]
    y_train = labels[inds_model]
    X_valid = feats_all[inds_valid]
    y_valid = labels[inds_valid]
    # split training set to folds, shuffle for stratification
    Xy = np.column_stack([X_train, y_train]) # merge X & y
    np.random.shuffle(Xy) # shuffle
    X_train = Xy[:, :-1] # split X
    y_train = Xy[:, -1] # split y
    y_train_enc = tf.keras.utils.to_categorical(y_train, num_classes=3) # Keras requires categorical outputs
    X_folds = np.array_split(X_train, n_members)
    y_folds = np.array_split(y_train_enc, n_members)
    
    # train ensemble
    preds_memb = [] # indiviual model predictions
    for i in range(n_members):
        # build model
        model = tf.keras.Sequential() # basic feed-forward
        model.add(tf.keras.layers.Dense( # every neuron connected
                n_neurons, # hidden layer neurons
                input_dim=3060, # input layer neurons
                activation='relu')) # hidden layer activation
        model.add(tf.layers.Dense(
                3, # output layer neurons
                activation='softmax')) # categorical, multi-class
        model.compile(
                # use custom learning rate, otherwise overfitting after 1st epoch
                optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                loss='categorical_crossentropy', # categorical
                metrics=['accuracy']) # score metrics
        
        # use other folds as test set
        X_test = []
        y_test = []
        for j in range(n_members):
            if j != i:
                X_test.extend(X_folds[j])
                y_test.extend(y_folds[j])
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        # train model with early stop
        model.fit(X_folds[i], y_folds[i], epochs=n_epochs, verbose=0,
                  validation_data=(X_test, y_test),
                  # define early stopping policy
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                          monitor='val_acc', # optimize validation accuracy
                          patience=25, # number of epochs to wait for better
                          verbose=0,
                          mode='max', # maximize accuracy
                          restore_best_weights=True)]) # restore optimum after wait
        # predict validation set
        preds_memb.append(model.predict_classes(X_valid))
    
    # generalize & save ensemble predictions
    preds_memb = np.array(preds_memb).T
    preds_fold = [] # generalized fold predictions
    for row in preds_memb:
        preds_fold.append(np.bincount(row).argmax())
    acc_valid.append(accuracy_score(y_valid, preds_fold))
    preds_mlp.extend(preds_fold)
    tf.keras.backend.clear_session() # clear GPU memory after ensemble
    print("fold-set complete")


# Evaluation:

print("\nacc-% (valid) per ensemble:")
print(acc_valid)
print("\nOverall accuracy-% (validation): "+ str(accuracy_score(labels_kf, preds_mlp)))
print(confusion_matrix(labels_kf, preds_mlp))


# Examples:

explot(inds_kf, preds_mlp, labels_kf, 0)
explot(inds_kf, preds_mlp, labels_kf, 1)
explot(inds_kf, preds_mlp, labels_kf, 2)
