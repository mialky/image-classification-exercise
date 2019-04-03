# -*- coding: utf-8 -*-

"""
Machine Learning and Pattern Recognition - Excercise Project

Created on Wed Mar 13 13:39:26 2019

@author: Mikael Kylanpaa
"""

############################## 1. Data import #################################

import numpy as np
from skimage import io


# Load images & assign labels (birdnests: 0, honeycombs: 1, lighthouses: 2):

imgs = []
labels = []

print("\nDownloading images")

# birdnests
for url in np.loadtxt('images/birdnests.txt', dtype='U100'):
    try:
        # add labels after succesful entries
        imgs.append(io.imread(url))
        labels.append(0)
    except:
        # skip faulty URLs
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


# Retrieve class indices:

labels = np.array(labels)
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



############################## 2. Data preparation ############################

import matplotlib.pyplot as plt
from statistics import mode
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte


# Store image sizes:

ws = []
hs = []

for img in imgs:
    ws.append(len(img))
    hs.append(len(img[0]))


# Resize images to mode w & h:

for i in range(len(imgs)):
    imgs[i] = resize(imgs[i], (mode(ws), mode(hs)))

# verify changes
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(imgs[0])
ax[0,1].imshow(imgs[1])
ax[1,0].imshow(imgs[inds2[0]])
ax[1,1].imshow(imgs[inds3[0]])
plt.show()


# Convert images to grayscale (for GLCM):

imgs_gr = []

for i in range(len(imgs)):
    imgs_gr.append(rgb2gray(imgs[i]))

# reduce quantization level (8-bit u-int)
for i in range(len(imgs_gr)):
    imgs_gr[i] = img_as_ubyte(imgs_gr[i])

plt.imshow(imgs_gr[0], cmap='Greys_r')
plt.show()



############################## 3. Feature extraction ##########################

from skimage.feature import greycomatrix, greycoprops
from scipy.stats import zscore


# Extract means & variances of RGB channels:

imgs = np.array(imgs)
feats_rgb = []

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
    # select some patches from middle
    patches = []
    patches.append(img[225:275, 225:275])
    patches.append(img[200:300, 200:300])
    patches.append(img[125:375, 125:375])
    
    feat_vect = []
    for patch in patches:
        # compute GLCM matrix from few angles
        glcm = greycomatrix(patch, distances=[5], angles=[0, np.pi/4, np.pi/2,
                            3*np.pi/4], levels=256, symmetric=True, normed=True)
        # compute & add GLCM values
        feat_vect.extend(greycoprops(glcm, 'contrast')[0][:])
        feat_vect.extend(greycoprops(glcm, 'dissimilarity')[0][:])
        feat_vect.extend(greycoprops(glcm, 'homogeneity')[0][:])
        feat_vect.extend(greycoprops(glcm, 'ASM')[0][:])
        feat_vect.extend(greycoprops(glcm, 'correlation')[0][:])
    
    feats_gr.append(feat_vect)

feats_gr = np.array(feats_gr)
print(feats_gr.shape)


# Standardize features by z-score:
feats_rgb = zscore(feats_rgb, axis=0)
feats_gr = zscore(feats_gr, axis=0)



############################# 4. Visualization - PCA #########################

from sklearn.decomposition import PCA


# Principal Component Analysis (2 PCs):

# RGB & GLCM separately
pca = PCA(n_components=2)
pca_rgb = pca.fit_transform(feats_rgb)
pca_gr = pca.fit_transform(feats_gr)

# RGB & GLCM combined
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



############################## 5. Visualization - SOM #########################

from minisom import MiniSom


# Using MiniSom library (https://github.com/JustGlowing/minisom):

# "if your dataset has 150 samples, 5*sqrt(150) = 61.23
# hence a map 8-by-8 should perform well."

# --> 54.5, will use 8x7 (=56) grid


# Train RGB SOM:

print("\nTraining SOM")
som = MiniSom(8, 7, len(feats_rgb[0]))
som.train_random(feats_rgb, 20000, verbose=True)

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



############################# 6. kNN classifier ##############################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score


# Reduce dimensionality with PCA:

pca = PCA(n_components=10) # 10 for RGB
pca_rgb = pca.fit_transform(feats_rgb)

pca = PCA(n_components=5) # 5 for GLCM
pca_gr = pca.fit_transform(feats_gr)

feats_knn = np.column_stack([pca_rgb, pca_gr]) # merge features
feats_knn = zscore(feats_knn, axis=0) # standardize again


## Use leave-one-out cross-validation to optimize k:
#
#ks = range(1, 26) # k candidates (1,2,...,25)
#acc = [] # k accuracies
#
#print("\nOptimizing kNN")
#
#for k in ks:
#    y_pred = [] # loo predictions
#    
#    for i in range(len(feats_knn)):
#        # assign test & train set
#        X_test = [feats_knn[i]]
#        X_train = np.delete(feats_knn, i, axis=0)
#        y_train = np.delete(labels, i)
#        # train classifier
#        knn = KNeighborsClassifier(n_neighbors=k)
#        knn.fit(X_train, y_train)
#        # predict test & store result
#        y_pred.append(knn.predict(X_test)[0])
#    
#    # calculate & store k accuracy
#    acc.append(accuracy_score(labels, y_pred))
#
#cv_result = np.column_stack([ks, acc])
#print(cv_result)


# Use nested cross-validation to optimize k:

ks = range(1, 26) # k candidates (1,2,...,25)
ks_best = [] # winner ks
acc_fold = [] # winner k fold-accuracies
acc_valid = [] # winner k validation accuracies

# use 6-fold cv for outer/validation loop (~20 samples/fold)
# 4 repeats --> 18 best ks to choose from
kf = RepeatedStratifiedKFold(n_splits=6, n_repeats=4, random_state=0)

print("\nOptimizing kNN (~1 min)")

for inds_model, inds_valid in kf.split(feats_knn, labels):
    # assign model training & validation sets
    X_model = feats_knn[inds_model]
    y_model = labels[inds_model]
    X_valid = feats_knn[inds_valid]
    y_valid = labels[inds_valid]
    
    # find best k for subset
    acc_ks = []
    for k in ks:
        y_pred = [] # loo predictions
        
        # use leave-one-out cv for inner/model training loop
        for i in range(len(X_model)):
            # assign test & train set
            X_test = [X_model[i]]
            X_train = np.delete(X_model, i, axis=0)
            y_train = np.delete(y_model, i)
            # train classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            # predict test & store result
            y_pred.append(knn.predict(X_test))
        
        # calculate & store k accuracy
        acc_ks.append(accuracy_score(y_model, y_pred))
    
    # find best k
    ind_best = np.argmax(acc_ks)
    k_best = ks[ind_best]
    ks_best.append(k_best)
    acc_fold.append(acc_ks[ind_best]) # store best k fold-accuracy
    # train whole fold with best k
    knn = KNeighborsClassifier(n_neighbors=k_best)
    knn.fit(X_model, y_model)
    # predict validation set
    y_pred = knn.predict(X_valid)
    acc_valid.append(accuracy_score(y_valid, y_pred))


# Calculate unique counts & averages:

ks_best = np.array(ks_best)
acc_valid = np.array(acc_valid)
cv_result = []

for k in np.unique(ks_best):
    cnt = np.count_nonzero(ks_best == k)
    inds = np.where(ks_best == k)
    acc = np.mean(acc_valid[inds])
    cv_result.append([k, cnt, acc])
    
cv_result = np.array(cv_result)

# sort array descending by k count
# (https://stackoverflow.com/a/2828121)
# (https://stackoverflow.com/a/16486305)
cv_result = cv_result[cv_result[:,1].argsort()[::-1]]
print(cv_result)
k_best = 6



############################# 7. Ridge classifier ############################

from sklearn.linear_model import RidgeClassifier


# Use all features with standardization:

feats_all = np.column_stack([feats_rgb, feats_gr])
feats_all = zscore(feats_all, axis=0)


# Use leave-one-out cross-validation to optimize alpha:

alphas = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000, 5000]
acc = []

print("\nOptimizing Ridge classifier")

for a in alphas:
    y_pred = []
    
    for i in range(len(feats_all)):
        X_test = [feats_all[i]]
        X_train = np.delete(feats_all, i, axis=0)
        y_train = np.delete(labels, i)
        ridge = RidgeClassifier(alpha=a)
        ridge.fit(X_train, y_train)
        y_pred.append(ridge.predict(X_test)[0])
    
    acc.append(accuracy_score(labels, y_pred))

cv_result = np.column_stack([alphas, acc])
print(cv_result)
a_best = 1000



############################# 8. Multi-layer perceptron ######################

import tensorflow as tf
from sklearn.model_selection import train_test_split


# Craft features for MLP committee/ensemble:

# assign seeds
np.random.seed(0)
tf.set_random_seed(0)

# split input to train & validation sets (use all 3060 features)
# (~ 20 validation images)
X_train, X_valid, y_train, y_valid = train_test_split(
        feats_all, labels, test_size=0.2, stratify=labels)

# apply one hot encoding
y_train_enc = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_valid_enc = tf.keras.utils.to_categorical(y_valid, num_classes=3)

# split train set to separate folds
n_members = 3 # use 3 folds/members (~ 30 images/member)
X_folds = np.array_split(X_train, n_members)
y_folds = np.array_split(y_train, n_members)
y_folds_enc = np.array_split(y_train_enc, n_members)


# Create MLP ensemble:

dim_h = 256 # hidden layer neurons
models = []
models = list()

for i in range(n_members):
    # define structure
    model = tf.keras.models.Sequential() # feed-forward
    model.add(tf.keras.layers.Dense(dim_h, # hidden layer dim
                                    input_dim=3060, # input dim
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(3, # output dim
                                    activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    # save model
    models.append(model)


# Train ensemble:

batch_size = 32 # inputs per weight update
n_epochs = 1000 # max whole set iterations

print(tf.keras.__version__)

print("\nOptimizing MLPs")

for i in range(n_members):
    print("\nMember " + str(i) + ":")
    # fit model
    models[i].fit(X_folds[i], y_folds_enc[i], batch_size, n_epochs, verbose=0,
              # find early stopping point
              callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', # value to optimize
                      patience=100, # checks after local minimum
                      verbose=1,
                      mode='min', # minimize/maximize
                      restore_best_weights=True)], # apply minimum
              validation_data=(X_valid, y_valid_enc))

## clear session/GPU memory
#tf.keras.backend.clear_session()



############################## 9. Accuracy estimation #########################

import pandas as pd
from sklearn.metrics import confusion_matrix


# Test optimized classifiers
# Use same test/validation split from MLP (already fit)

# kNN:

# concatenate for PCA & standardization
X_knn = np.concatenate((X_train, X_valid), axis=0)
pca = PCA(n_components=10)
X_knn = pca.fit_transform(X_knn)
X_knn = zscore(X_knn, axis=0)

# split again
X_train_knn = X_knn[:len(X_train)]
X_valid_knn = X_knn[-len(X_valid):]

# train & evaluate
knn = KNeighborsClassifier(n_neighbors=k_best)
knn.fit(X_train_knn, y_train)
preds_knn = knn.predict(X_valid_knn)


# Ridge classifier:

ridge = RidgeClassifier(alpha=1000)
ridge.fit(X_train, y_train)
preds_ridge = ridge.predict(X_valid)


# MLP committee:

preds_mlp = []

for i in range(len(X_valid)):
    inst = np.expand_dims(X_valid[i], axis=0) # single input to correct format
    preds = [] # instance predictions
    
    for j in range(n_members):
        preds.append(models[j].predict(inst))
    means = np.mean(preds, axis=0) # calculate ensemble mean
    preds_mlp.append(np.argmax(means)) # store ensemble prediction

preds_mlp = np.array(preds_mlp)


# Evaluate models:

print("\nPredictions")
preds = pd.DataFrame(np.column_stack([y_valid, preds_knn, preds_ridge, preds_mlp]))
preds.columns = ['true', 'knn', 'ridge', 'mlp']
print(preds.to_string(index=False))

print("\nAccuracies")
print("knn: " + str(accuracy_score(y_valid, preds_knn)))
print("ridge: " + str(accuracy_score(y_valid, preds_ridge)))
print("mlp: " + str(accuracy_score(y_valid, preds_mlp)))

print("\nConfusion matrix")
print("knn:\n", confusion_matrix(y_valid, preds_knn))
print("ridge:\n", confusion_matrix(y_valid, preds_ridge))
print("mlp:\n", confusion_matrix(y_valid, preds_mlp))


############################## 10. Cross validation ###########################

from sklearn.model_selection import StratifiedKFold


# Use 6-fold cross-validation on all models (~20 samples/fold):

kf = StratifiedKFold(n_splits=6)
y_real = []
y_knn = []
y_ridge = []
y_mlp = []

print("\nEstimating accuracy")

for inds_model, inds_valid in kf.split(feats_knn, labels):
    # assign train & validation sets
    X_train = feats_all[inds_model]
    y_train = labels[inds_model]
    X_valid = feats_all[inds_valid]
    y_valid = labels[inds_valid]
    X_train_knn = feats_knn[inds_model]
    X_valid_knn = feats_knn[inds_valid]
    y_train_enc = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_valid_enc = tf.keras.utils.to_categorical(y_valid, num_classes=3)
    # save real
    y_real.append(y_valid)
    # train knn
    knn = KNeighborsClassifier(n_neighbors=k_best)
    knn.fit(X_train_knn, y_train)
    y_knn.append(knn.predict(X_valid_knn))
    # train ridge
    ridge = RidgeClassifier(alpha=a_best)
    ridge.fit(X_train, y_train)
    y_ridge.append(ridge.predict(X_valid))
    
    # Train single mlp instead of committee/ensemble:
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(dim_h, input_dim=3060, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train_enc, epochs=100, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=100, verbose=1,
                      mode='min', restore_best_weights=True)],
              validation_data=(X_valid, y_valid_enc))
    
    y_mlp.append(model.predict_classes(X_valid_enc))




