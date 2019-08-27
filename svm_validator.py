# whole code as .py script to be run from command line

#####################################################################
# The best performance was obtained with linear, ploy and rbf. As a
# result, linear has been selected as default in kernel option. To run
# with default from command line use 'python filename.py'. To choose
# other kernels, 'python filename.py -k poly'. Available kernels are
# 'linear','poly','rbf','sigmoid'. The random state has been set to 0
# for reproducibility.
#####################################################################

# import modules
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import io

# add optional command line argument for selecting kernel
parser = argparse.ArgumentParser()
parser.add_argument('-k', default='linear',
                    help="selects the kernel. options are: 'linear','poly','rbf','sigmoid' (default is linear)")
args = parser.parse_args()
kernel = args.k  # read the dataset
raw_data = pd.read_csv("generated_data.csv")

# make  alist of validation text files
validation_file_list = []
for root, dirs, files in os.walk("validation folder/"):
    validation_file_list = files

# seperating predictor & target variable
X = raw_data.drop('type', axis=1)
y = raw_data['type']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# kernel_option=['linear','poly','rbf','sigmoid']

# define the function for classification along with metrics

def classify(kernel):
    svclassifier = SVC(kernel=kernel, random_state=0)
    svclassifier.fit(X_train, y_train)

    return svclassifier


# for validation text files stored in valodation folder
def validate(svclassifier, validation_file_list):
    for i in validation_file_list:
        print(i[2:])
        with io.open("validation folder/" + i[2:], "r", encoding="utf-8") as my_file:
            my_unicode_string = my_file.read()
            d = json.loads(my_unicode_string)
            single_pred = svclassifier.predict(np.array(list(d.values())).reshape(1, -1))
            #print("Predictors\n", d)
            #print("Predicted Class: ", single_pred)
            if single_pred == 1:
                print(i, ": error")


x = classify(kernel=kernel)
validate(x, validation_file_list)
