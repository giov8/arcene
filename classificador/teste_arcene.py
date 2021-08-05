from __future__ import print_function
import numpy as np
#import os
from sklearn import svm, metrics
from genetic_selection import GeneticSelectionCV


def main():
    # Load dataset
    f = open('./../dataset/arcene_train.data')
    X_train = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()
    X_train.resize(100,10000) # reshape data as (n_samples, n_features)

    f = open('./../dataset/arcene_train.labels')
    y_train = np.fromfile(f, dtype=np.int32, sep=' ')
    f.close()

    f = open('./../dataset/arcene_valid.data')
    X_test = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()
    X_test.resize(100, 10000) # reshape data as (n_samples, n_features)

    f = open('./../dataset/arcene_valid.labels')
    y_test = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    print(X_train.shape)
    print(y_train.size)
    print(X_test.size)
    print(y_test.size)

    print("Dados Carregados")

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Select features using genetic algorithm
    selector = GeneticSelectionCV(clf,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=1000,
                                  n_population=100,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=1000,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=100,
                                  caching=True,
                                  n_jobs=-1)

    #Train the model using the training sets
    selector.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = selector.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred))


if __name__ == "__main__":
    main()  
