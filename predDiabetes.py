import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sns.set_style('darkgrid')


def main():
    dataset = pd.read_csv('diabetes_csv.csv')

    # replacing strings in outcome to binary
    dataset['Class'] = dataset['Class'].replace('tested_negative', int(0))
    dataset['Class'] = dataset['Class'].replace('tested_positive', int(1))

    # replace 0 (no data) in these columns with avg person data
    no_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for col in no_zeroes:
        dataset[col] = dataset[col].replace(0, np.NaN)
        mean = int(dataset[col].mean(skipna=True))
        dataset[col] = dataset[col].replace(np.NaN, mean)

    # plotting
    sns.countplot(x=dataset['Class'], data=dataset)
    sns.displot(dataset['Pregnancies'])
    sns.displot(dataset['Glucose'])
    sns.displot(dataset['BloodPressure'])
    sns.displot(dataset['SkinThickness'])
    sns.displot(dataset['Insulin'])
    sns.displot(dataset['BMI'])
    sns.displot(dataset['DiabetesPedigreeFunction'])
    sns.displot(dataset['Age'])
    sns.pairplot(dataset)
    plt.show()

    # checking correlations
    matrix = dataset.corr()
    ax = plt.subplots(figsize=(9, 6)), sns.heatmap(matrix, vmax=0.8, square=True, cmap="coolwarm")
    plt.show()

    # splitting the dataset
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']

    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # feature scaling
    # acc up to 0.(81) from 0.(76)
    # f1 up to 0.69565 from 0.(62)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # building the model with KNN algorithm
    # this method is the most optimal
    # changing k manually to lower or higher value results in worse acc and f1
    k = int(math.sqrt(len(y_test)))
    if k % 2 == 0:
        k -= 1
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')

    # fitting the model
    knn.fit(X_train, y_train)

    # prediction
    y_pred = knn.predict(X_test)

    # accuracy
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy: ", format(metrics.accuracy_score(y_test, y_pred)))
    print("F1 score:", format(f1_score(y_test, y_pred)))

    # print(X_test)

    # print(knn.predict_proba(X_test))


if __name__ == '__main__':
    main()
