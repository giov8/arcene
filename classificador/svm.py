import numpy as np
#import os
from sklearn import svm, metrics


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


    print("Classificação com SVM (sem seleção de features)")

    for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
        
        if (kernel=='poly'):
            for degree in range(2,8):
                #Create a svm Classifier
                clf = svm.SVC(kernel=kernel, degree=degree) # Linear Kernel

                #Train the model using the training sets
                clf.fit(X_train, y_train)

                #Predict the response for test dataset
                y_pred = clf.predict(X_test)

                print()
                print("Kernel: ",kernel, "Degree: ",degree)

                # Model Accuracy: how often is the classifier correct?
                print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
                # Model Precision: what percentage of positive tuples are labeled as such?
                print("Precision:",metrics.precision_score(y_test, y_pred))
                # Model Recall: what percentage of positive tuples are labelled as such?
                print("Recall:",metrics.recall_score(y_test, y_pred))

        else:
            #Create a svm Classifier
            clf = svm.SVC(kernel=kernel)
            #clf = svm.SVC(kernel=kernel, degree=5) # Linear Kernel

            #Train the model using the training sets
            clf.fit(X_train, y_train)

            #Predict the response for test dataset
            y_pred = clf.predict(X_test)

            print()
            print("Kernel: ",kernel)

            # Model Accuracy: how often is the classifier correct?
            print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            # Model Precision: what percentage of positive tuples are labeled as such?
            print("Precision:",metrics.precision_score(y_test, y_pred))
            # Model Recall: what percentage of positive tuples are labelled as such?
            print("Recall:",metrics.recall_score(y_test, y_pred))


if __name__ == "__main__":
    main()  
