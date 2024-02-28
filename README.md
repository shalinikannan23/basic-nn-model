# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1) Developing a Neural Network Regression Model AIM To develop a neural network regression model for the given dataset. THEORY Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it.

2) Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.

3) First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://github.com/shalinikannan23/basic-nn-model/assets/118656529/a0bf58cd-cb44-4965-88f8-dcdfe335bb41)


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
df=df.astype({'Input':'float'})
df=df.astype({'Output':'float'})
X=df[['Input']].values
Y=df[['Output']].values
```

## Split the testing and training data
```py
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
scaler=MinMaxScaler()1
scaler.fit(x_train)
x_t_scaled = scaler.transform(x_train)
x_t_scaled
```

## Build the Deep learning Model
```py
ai_brain = Sequential([
    Dense(3,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_t_scaled,y=y_train,epochs=50)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
scal_x_test=scaler.transform(x_test)
ai_brain.evaluate(scal_x_test,y_test)
input=[[120]]
inp_scale=scaler.transform(input)
inp_scale.shape
ai_brain.predict(inp_scale)
```

## Dataset Information

![image](https://github.com/shalinikannan23/basic-nn-model/assets/118656529/45ef9a96-d6e3-4ae0-a74e-302bac0f4782)


## OUTPUT

## Training Loss Vs Iteration Plot
![image](https://github.com/shalinikannan23/basic-nn-model/assets/118656529/98a585b8-107d-44cf-b645-d59104497783)


## Test Data Root Mean Squared Error

![image](https://github.com/shalinikannan23/basic-nn-model/assets/118656529/3d6bcb0d-2441-4d40-8c41-19b96ff2724e)


## New Sample Data Prediction

![image](https://github.com/shalinikannan23/basic-nn-model/assets/118656529/0dfe0727-e9d2-4056-bc30-0e7ceceba332)


## RESULT
Thus a Neural network for Regression model is Implemented
