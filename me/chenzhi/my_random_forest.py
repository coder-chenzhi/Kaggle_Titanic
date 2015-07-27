__author__ = 'chenzhi'
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd


def has_nan(data):
    for v in data:
        if str(v) == "nan":
            return True
    return False


if __name__ == "__main__":
    loc_train = "G:\\KuaiPan\\Project\\Kaggle\\Titanic\\data\\train.csv"
    loc_test = "G:\\KuaiPan\\Project\\Kaggle\\Titanic\\data\\test.csv"
    loc_submission = "G:\\KuaiPan\\Project\\Kaggle\\Titanic\\data\\kaggle.forest.submission.csv"

    df_train = pd.read_csv(loc_train)
    df_test = pd.read_csv(loc_test)

    # pre-process
    # fill Nan
    df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())
    df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean())
    df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].mean())
    # change Sex from string to int
    le = preprocessing.LabelEncoder()
    le.fit(["male", "female"])
    df_train["Sex"] = le.transform(df_train["Sex"])
    df_test["Sex"] = le.transform(df_test["Sex"])


    feature_cols = ["Pclass", "Sex", "Age"]

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y = df_train['Survived']
    test_ids = df_test['PassengerId']

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score=True, random_state=42, max_features=None,
                                 min_samples_leaf=10)

    clf.fit(X_train, y)

    count = 0.0
    for i, v in enumerate(clf.predict(X_train)):
        if v == y[i]:
            count += 1
    print "precision", count / len(y)
    with open(loc_submission, "wb") as outfile:
        outfile.write("PassengerId,Survived\n")
        for e, val in enumerate(list(clf.predict(X_test))):
            outfile.write("%s,%s\n" % (test_ids[e], val))

