import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data
target = iris.target

df = pd.concat([pd.DataFrame(data=features,columns=iris.feature_names),
           pd.DataFrame(data=target,columns=['target'])],
           axis=1)
