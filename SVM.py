import os
import cv2
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

# Prepare/collect data
path = os.listdir('brain_tumor/Training/')
classes = {'no_tumor': 0, 'pituitary_tumor': 1}

X = []
Y = []

for cls in classes:
    pth = 'brain_tumor/Training/' + cls
    for j in os.listdir(pth):
        img = cv2.imread(pth + '/' + j, 0)
        img = cv2.resize(img, (200, 200))
        X.append(img)
        Y.append(classes[cls])

# Reshape data
X = np.array(X)
Y = np.array(Y)
X_updated = X.reshape(len(X), -1)
# Visualize data
plt.imshow(X[0], cmap='gray')
# Split Data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=0.20)

# Feature Scaling
xtrain = xtrain / 255
xtest = xtest / 255

# Feature Selection: PCA
pca = PCA(0.98)
pca.fit_transform(xtrain)
pca.transform(xtest)

# Train Model
sv = SVC()
sv.fit(xtrain, ytrain)

# Evaluation

print("Training Score (SVM):", sv.score(xtrain, ytrain))
print("Testing Score (SVM):", sv.score(xtest, ytest))

# Prediction for SVM
pred = sv.predict(xtest)
misclassified = np.where(ytest != pred)
print("Total Misclassified Samples(SVM): ", len(misclassified[0]))

# TEST MODEL
dec = {0: 'No Tumor', 1: 'Positive Tumor'}

# Accuracy
sv_pred = sv.predict(xtest)
sv_accuracy = accuracy_score(ytest, sv_pred)

# Precision
sv_precision = precision_score(ytest, sv_pred)

# Initialize misclassification counters
misclassified_svm = 0

# Visualize SVM predictions on 'no_tumor' images
plt.figure(figsize=(12, 12))
plt.suptitle('Support Vector Machine Predictions - No Tumor')

# Testing SVM on 'no_tumor'
for c, i in enumerate(os.listdir('brain_tumor/Testing/no_tumor/')[:20], 1):
    plt.subplot(4, 5, c)  # Use 4 rows and 5 columns for a 20-image display
    img = cv2.imread('brain_tumor/Testing/no_tumor/' + i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1) / 255
    prediction = sv.predict(img1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{dec[prediction[0]]}")
    plt.axis('off')
    if prediction[0] != 0:  # Misclassified
        misclassified_svm += 1

# Visualize SVM predictions on 'pituitary_tumor' images
plt.figure(figsize=(12, 12))
plt.suptitle('Support Vector Machine Predictions - Pituitary Tumor')

# Testing SVM on 'pituitary_tumor'
for c, i in enumerate(os.listdir('brain_tumor/Testing/pituitary_tumor/')[:20], 1):
    plt.subplot(4, 5, c)  # Use 4 rows and 5 columns for a 20-image display
    img = cv2.imread('brain_tumor/Testing/pituitary_tumor/' + i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1) / 255
    prediction = sv.predict(img1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{dec[prediction[0]]}")
    plt.axis('off')
    if prediction[0] != 1:  # Misclassified
        misclassified_svm += 1

plt.show()
# Print total misclassified samples for each model
print("Total Misclassified Samples (Support Vector Machine):", misclassified_svm)

# Prediction for SVM
pred_prob = sv.decision_function(xtest)  # Decision function scores for binary classification
fpr, tpr, thresholds = roc_curve(ytest, pred_prob)
roc_auc = roc_auc_score(ytest, pred_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Support Vector Machine')
plt.legend(loc='lower right')
plt.show()
