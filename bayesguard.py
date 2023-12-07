# =============================================================================
# Project Name: BayesGuard
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Reading the dataset
data = pd.read_csv("emails.csv")
data.drop(columns=['Email No.'], inplace = True)

# total of the number of empty rows
data.isna().sum() 

#correlation between columns and spam/not spam pos = more likely to be spam
corr_matrix = data.corr()
print(corr_matrix["Prediction"].sort_values(ascending = False))

#data.shape
x = data.iloc[: ,0:3000]  
y = data.iloc[: , 3000]

#(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)  # 80% training, 20% test 

# Gaussian Naive Bayes
model = GaussianNB() 
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

# Counting spam and non-spam emails
spam_count = np.sum(y_prediction == 1) 
legit_count = np.sum(y_prediction == 0) 
training_size = spam_count + legit_count

# Output
print("===== BayesGuard ===================")
print("- Accuracy : ", np.round(accuracy_score(y_test, y_prediction), 4) * 100)
print("- Cross-Validation score : ", np.round(cross_val_score(model, x, y, cv=5, scoring="accuracy").mean(), 4) * 100)
print("====================================")
print("Training size : ", training_size)
print("- Spam emails:", spam_count)
print("- Legit emails:", legit_count)
print("====================================")
# =============================================================================
