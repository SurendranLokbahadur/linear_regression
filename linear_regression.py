
# importing the libraries

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# loading the dataset

df = pd.read_csv('insurance.csv')


plt.title('age vs insurance-Charges (before linear-regression)') # providing the title the scatterplot
sns.scatterplot(data = df, x ='age', y = 'charges') # displaying the scatterploat as age vs charges
plt.show()


# fitting as Linear-regression model to the chosen-data

x = df[['age']].values
y = df['charges'].values
reg = LinearRegression()

model = reg.fit(x,y)
linreg_model = LinearRegression()
linreg_model.fit(x, y)

# predicting the data
y_pred = linreg_model.predict(x)

plt.figure()
sns.scatterplot(data=df, x='age', y='charges') # displaying the scatterploat as age vs charges

plt.plot(x, y_pred , color = 'orange')   # plotting the best-fit line
plt.title('age vs insurance-Charges (after linear-regression)') # giving the title
plt.show()
plt.close()