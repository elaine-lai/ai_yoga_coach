import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def supportvectorvclassifier(X_train, X_test, y_train, y_test, show_csv):
    svm_classifier = SVC(kernel='rbf', C=10, probability=True)
    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"SVC Accuracy: {accuracy}")
    print("SVC Classification Report:")
    print(classification_report(y_test, predictions))
    if show_csv:
        result_df = pd.DataFrame({'Prediction': predictions, 'Actual': y_test})
        result_df.to_csv('../generated_csv/svc.csv', index=False)
    filename = '../model/svc_model.joblib'
    joblib.dump(svm_classifier, filename)


def randomforestclassifier(X_train, X_test, y_train, y_test, show_csv):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"RFC Accuracy: {accuracy}")
    print("RFC Classification Report:")
    print(classification_report(y_test, predictions))
    if show_csv:
        result_df = pd.DataFrame({'Prediction': predictions, 'Actual': y_test})
        result_df.to_csv('../generated_csv/rfc.csv', index=False)


def kneighborsclassifier(X_train, X_test, y_train, y_test, show_csv):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"KNN Accuracy: {accuracy}")
    print("KNN Classification Report:")
    print(classification_report(y_test, predictions))
    if show_csv:
        result_df = pd.DataFrame({'Prediction': predictions, 'Actual': y_test})
        result_df.to_csv('../generated_csv/knn.csv', index=False)


def load_data(file_path):
    data = pd.read_csv(file_path)

    # Shuffle the DataFrame (since data/poseangles.generated_csv has each pose sequentially)
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = shuffled_data.drop("pose", axis=1)
    y = shuffled_data["pose"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
