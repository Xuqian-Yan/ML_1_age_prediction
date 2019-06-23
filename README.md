# The original project description from https://www.kaggle.com/c/ml-project-1/overview

# Description

In this project you apply regression techniques to predict a person's age from a three-dimensional magnetic resonance image of their brain.

Magnetic Resonance Imaging (MRI) is a key technology in medical imaging as it provides a non-invasive and non-radiative tool to investigate sensitive organs such as the brain. MRI typically operates on a medium length scale which corresponds to a resolution of about 1mm³. Nevertheless, the three-dimensional structure, small sample size and individual brain shapes make it difficult to recognize disease related patterns, even for skilled neurologists.

Considerable research effort is invested to learn complex disease patterns automatically. This would not only solve a difficult technical problem, but it would also significantly support the treatment of people suffering from neurodegenerative diseases.

In this project we start with a more modest goal and try to predict the age of a person from their brain MR image. You will learn how to deal with real medical image data, and how to process it to apply machine learning techniques successfully.

# Evaluation

The evaluation metric for this competition is Mean-Squared-Error (MSE). The MSE score, a basic measure of fit, represents the average deviation of n predictions ŷ i from their true values yi. The Mean-Squared-Error is given by:

MSE=1n∑i=1n(ŷ i−yi)2
The MSE metric weights large deviations much heavier than small deviations. Consequently, it is particularly vulnerable to outliers.

# Data

X_train.npy - the training set provided as numpy array with shape (278, 6443008). The rows run over samples, the columns over features. Basically, each feature is a different voxel (3D pixel) of the image. You can get the 3D structure back with numpy. reshape(X_train, (-1, 176, 208, 176)).

X_test.npy - the test set provided in the same format as the training set, but with shape (138, 6443008).

y_1.csv - The targets for the regressions training. The k-th row contains the age of the k-th sample in X_train.npy.

sampleSubmission.csv - a sample submission file in the correct format. The project framework saves the correct format, but we also provide a description here. Your submissions should have two columns and 139 = 1+138 rows (one row for headers and then one row for each test sample, no more and no less). The first column is headed by "ID" and then simply runs from 1 to 138, the second column is headed by "Prediction" where any consequent row k+1 contains your prediction for the k-th sample of X_test.npy.
