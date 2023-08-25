import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'caab573c-c37d-49a0-8342-6a8913be7890'
resource_group = 'mlops-aug-batch'
workspace_name = 'Intellipat-mlops'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='Iris')
df = dataset.to_pandas_dataframe()




#def_test_columns():
#   assert df.columns.to_list() = ['SepalLengthCm','SepalWidthCm','PetalLengthCm',#'PetalWidthCm']

def_test_accuracy():
    features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    target = 'Species'
    X_train, X_test, y_train, y_test = train_test_split(df[features],
    df[target], test_size=0.1,shuffle=True)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)  
    assert  accuracy_score(y_test,y_pred) > 0.90 