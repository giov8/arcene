from __future__ import print_function
import numpy as np
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

    filename = 'teste3.csv'
    print("max_features,n_population,n_generations,accuracy,precision,recall", file=open(filename, 'a'))

    #Define the tests range
    max_feat = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, None]
    n_pop = [100, 200, 500, 1000, 200, 100, 50, 10, 20, 50, 100]
    n_gen =[1000, 500, 200, 100, 50, 100, 200, 100, 50, 20, 10]

    #Test loop
    for max_features in max_feat:
        for i in range (11):
            print("Max features:", max_features, "N população", n_pop[i], "N geração", n_gen[i])

            #Create a svm Classifier
            clf = svm.SVC(kernel='poly', degree=5) # Polynomial Kernel, degree 5

            #Select features using genetic algorithm
            selector = GeneticSelectionCV(clf,
                                          verbose=1,
                                          scoring="accuracy",
                                          max_features=max_features,
                                          n_population=n_pop[i],
                                          crossover_proba=0.5,
                                          mutation_proba=0.2,
                                          n_generations=n_gen[i],
                                          tournament_size=5,
                                          n_gen_no_change=100,
                                          caching=True,
                                          n_jobs=-1)

            #Train the model using the training sets
            selector.fit(X_train, y_train)

            #Predict the response for test dataset
            y_pred = selector.predict(X_test)

            with open(filename, "a") as f:
                print(max_features, n_pop[i], n_gen[i], metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), file=f, sep=',')


if __name__ == "__main__":
    main()
