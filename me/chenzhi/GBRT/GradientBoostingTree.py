__author__ = 'chenzhi'

import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

loc_train = "G:\\KuaiPan\\Project\\Kaggle\\Titanic\\commit\\9\\train_add_title.csv"
loc_test = "G:\\KuaiPan\\Project\\Kaggle\\Titanic\\commit\\9\\test_add_title.csv"
loc_submission = "G:\\KuaiPan\\Project\\Kaggle\\Titanic\\data\\kaggle.forest.submission.csv"

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv(loc_train, header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# add by coder_chenzhi
Titles = list(enumerate(np.unique(train_df['Title'])))    # determine all values of Embarked,
Titles_dict = { name : i for i, name in Titles }              # set up a dictionary in the form  Ports : index
train_df.Title = train_df.Title.map( lambda x: Titles_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)


# TEST DATA
test_df = pd.read_csv(loc_test, header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# add by coder_chenzhi
test_df.Title = test_df.Title.map( lambda x: Titles_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId',], axis=1)

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
original_params = {'n_estimators': 500, 'max_leaf_nodes': 4, 'random_state': 17,
                   'subsample': 0.7, 'min_samples_split': 50}
params = dict(original_params)
clf = GradientBoostingClassifier(**params)
clf = clf.fit( train_data[0::,1::], train_data[0::,0] )
importances = clf.feature_importances_

print "Score on train data", clf.score(train_data[0::,1::], train_data[0::,0])

print 'Predicting...'
output = clf.predict(test_data).astype(int)


predictions_file = open(loc_submission, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender', 'Title']
for zipped in zip(features, importances):
    print "features %s importances: %s" % (zipped[0], zipped[1])

plt.figure()
plt.title("Feature importances")
plt.bar(range(len(features)), importances, color="r", align="center")
plt.xticks(range(len(features)), features)
plt.xlim([-1, len(features)])
plt.show()

print 'Done.'

