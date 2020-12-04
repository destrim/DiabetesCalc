import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def main():
    dataset = pd.read_csv('diabetes_csv.csv')
    print(len(dataset))

    # replacing strings in outcome to binary
    dataset['Class'] = dataset['Class'].replace('tested_negative', int(0))
    dataset['Class'] = dataset['Class'].replace('tested_positive', int(1))

    # replace 0 (no data) in these columns with avg person data
    no_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for col in no_zeroes:
        dataset[col] = dataset[col].replace(0, np.NaN)
        mean = int(dataset[col].mean(skipna=True))
        dataset[col] = dataset[col].replace(np.NaN, mean)

    # split dataset
    X = dataset.iloc[:, 0:8]
    y = dataset.iloc[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # feature scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # init K-NN
    k = int(math.sqrt(len(y_test)))
    if k % 2 == 0:
        k -= 1
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')

    # fit model
    classifier.fit(X_train, y_train)

    # predict the test set results
    y_pred = classifier.predict(X_test)

    # evaluate model
    print(confusion_matrix(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
