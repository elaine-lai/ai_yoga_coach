from select_classifier import *

file_path = "../data/pose_angles.csv"
X_train, X_test, y_train, y_test = load_data(file_path)


# choices are: rfc, svm, knn
show_csv = False
randomforestclassifier(X_train, X_test, y_train, y_test, show_csv)
supportvectorvclassifier(X_train, X_test, y_train, y_test, show_csv)
kneighborsclassifier(X_train, X_test, y_train, y_test, show_csv)

'''
best one looks like its SVC, with accuracy of 0.9694835680751174
'''