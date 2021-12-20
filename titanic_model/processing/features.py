import numpy as np

def preprocessing(data):
    # was the person alone
    data = data.copy()
    data['alone'] = np.where((data['SibSp']==0) & (data['Parch']==0), 1, 0)

    # family member total
    data['family'] = data['SibSp'] + data['Parch']

    # was the person a baby
    data['is_baby'] = np.where(data['Age'] < 5, 1, 0)

    # create a title column
    data['title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    data['title'] = data['title'].replace('Mlle', 'Miss')
    data['title'] = data['title'].replace('Ms', 'Miss')
    data['title'] = data['title'].replace('Mme', 'Mrs')
    data['title'] = data['title'].replace('Don', 'Mr')
    data['title'] = data['title'].replace('Dona', 'Mrs')

    # drop data not needed anymore
    data.drop(['SibSp', 'Parch', 'Name'], axis=1, inplace=True)

    return data