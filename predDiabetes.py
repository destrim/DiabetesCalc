import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

    print(dataset.info())

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # building the model with KNN algorithm
    k = int(math.sqrt(len(y_test)))
    if k % 2 == 0:
        k -= 1
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')

    # fitting the model
    knn.fit(X_train, y_train)

    # prediction
    y_pred = knn.predict(X_test)

    # accuracy
    print("Accuracy: ", format(metrics.accuracy_score(y_test, y_pred)))

    print(X_test)

    print(knn.predict_proba(X_test))


if __name__ == '__main__':
    main()
