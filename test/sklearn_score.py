from args import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree

# arg = EPS5_Arguments()
arg = DCC_Arguments()
# arg = BCW_Arguments()

x_train, x_test, y_train, y_test = train_test_split(arg.data_x, arg.data_y, test_size=.3)

stand1 = StandardScaler()
stand1.fit(x_train)
x_train_standard = stand1.transform(x_train)

stand2 = StandardScaler()
stand2.fit(x_test)
x_test_standard = stand1.transform(x_test)

lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train_standard, y_train.ravel())
y_hat = lr.predict(x_test_standard)
print('Logistic Regression: \nf1-score:{:.3}\nacc:{:.3} \n\n'
      .format(accuracy_score(y_test.ravel(), y_hat), f1_score(y_test.ravel(), y_hat)))


clf = tree.DecisionTreeClassifier(class_weight='balanced')
clf = clf.fit(x_train_standard, y_train.ravel())
y_hat = clf.predict(x_test_standard)
print('Decision Tree: \nf1-score:{:.3}\nacc:{:.3} \n'
      .format(accuracy_score(y_test.ravel(), y_hat), f1_score(y_test.ravel(), y_hat)))
