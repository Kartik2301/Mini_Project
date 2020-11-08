import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('data_set.csv', index_col=0)

X = data.drop(['label','subject'], axis=1).values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


classifier = LinearRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(classifier.score(X_test,y_test))


classifier = RandomForestClassifier(max_depth=4, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(str(accuracy_score(y_test, y_pred)))