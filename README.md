# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1) Developing a Neural Network Regression Model AIM To develop a neural network regression model for the given dataset. THEORY Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it.

2) Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.

3) First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

<img height=10% width=30% src ="https://github.com/shalinikannan23/basic-nn-model/assets/118656529/6a9ad24d-d72b-4572-88db-3214cbbcef64">



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM


## 212222240095
## SHALINI K(AIML)
## Importing Required packages
```py
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd
```

## Authenticate the Google sheet
```py
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Data').sheet1
```

## Construct Data frame using Rows and columns
```py
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
X=df[['Input']].values
Y=df[['Output']].values
```

## Split the testing and training data
```py
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=40)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
X_train1 = Scaler.transform(x_train)
```

## Build the Deep learning Model
```py
ai_brain=Sequential([
    Dense(9,activation = 'relu',input_shape=[1]),
    Dense(16,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer='adam',loss='mse')
ai_brain.fit(X_train1,y_train.astype(np.float32),epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
test=Scaler.transform(x_test)
ai_brain.evaluate(test,y_test.astype(np.float32))
n1=[[40]]
n1_1=Scaler.transform(n1)
ai_brain.predict(n1_1)
```

## Dataset Information

<img height=50% width=30% src ="https://github.com/shalinikannan23/basic-nn-model/assets/118656529/a0610a1a-b27c-48ae-a4fd-3ba4758cdb1c">


## OUTPUT

## Training Loss Vs Iteration Plot
<img height=10% width=30% src ="https://github.com/shalinikannan23/basic-nn-model/assets/118656529/8613b556-1cf1-4f92-afeb-2a01beb28550">


## Test Data Root Mean Squared Error

<img height=10% width=30% src ="https://github.com/shalinikannan23/basic-nn-model/assets/118656529/408fcfce-0e2d-422e-8a9a-46cd80b7c610">


## New Sample Data Prediction

<img height=10% width=30% src ="https://github.com/shalinikannan23/basic-nn-model/assets/118656529/926eb56e-0d04-4201-a187-5617e9448799">



## RESULT
Thus a Neural network for Regression model is Implemented
