# Machine Learning Project
# Objective :

You are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for the management to
understand the pricing dynamics of a new market.


Dataset used : 

https://drive.google.com/file/d/1FHmYNLs9v0Enc-UExEMpitOFGsWvB2dP/view?usp=drive_link

# Loading Dataset

Libraries imported

import pandas as pd
import numpy as np

Code : 

df=pd.read_csv("CarPrice_Assignment.csv")

![image](https://github.com/user-attachments/assets/dde549ab-fb45-4853-94eb-587696dae33c)

The dataset have 205 rows and 26 columns.

Columns are 'car_ID', 'symboling', 'CarName', 'fueltype', 'aspiration',
       'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
       'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'price'

Datatypes of each columns are as follows :

![image](https://github.com/user-attachments/assets/b62b888b-086e-4b93-ac98-40afc89590bb)

Histplot is drawn to analyse the distribution of data and check for outliers. Some of them are shown below :

![image](https://github.com/user-attachments/assets/0a10b230-10a3-492d-9aed-b7563b551477)

Boxplot is shown to check for outliers in each columns. Some of the columns having outliers are shown below :

![image](https://github.com/user-attachments/assets/a560cd6f-d438-40d3-8cc3-4e715ee7a332)

# Data Preprocessing

Outliers are found in carwidth, enginesize, stroke, compressionratio, horsepowerhighwaympg and price.

Correlation with heatmap is shown to interpret the relation and multicollinearity.

Columns which are highly correlated with target variable - price :

* highwaympg
* citympg
* horsepower
* enginesize
* curbweight
* carwidth

Columns with multicollinearity :

* highwaympg and citympg
* carlength and curbweight
* carwidth and curbweight
* wheelbase and carlength
* enginesize and curbweight

Scatter plots are drawn to ensure multicollinearity of columns. 

Columns with multicollinearity are removed. Columns which are irrelevant and which doesnot give any information are dropped.

Columns which have multicollinearity are citympg, curbweight and wheelbase. car_id irrelevant. It is also removed.

For handling outliers, IQR method is used.

![image](https://github.com/user-attachments/assets/ea53d4f7-a9bc-42a0-955c-f21a0a016a4b)

# Feature Engineering

* Define feature variables and targetted variables
* Identify categorical and numerical features
* Encoding categorical features by using Label Encoding
* Apply Standard scaling to the numerical features

# Model Implementation

* Split data into training and testing sets
* Model Selection
  ![image](https://github.com/user-attachments/assets/b6ea998a-065f-420f-b1d4-b754b9eea02e)

  Compare the performance of all the models based on R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

  ![image](https://github.com/user-attachments/assets/9913f670-0bbd-4cf7-88ac-cb96b23ac2d9)

  RandomForestRegressor has the best performance among all listed models.
Random Forest gives lowest RMSE(14.48), MSE(209.8), and MAE(11.5), and the highest R² score(92.16). It suggests that the Random Forest Regressor is able to capture the underlying patterns in the data very well. Model which is performed least is SVM with lowest R2 square and highest RMSE, MSE and RAE.

# Feature Selection

Feature selection is done using methods like SelectKBest, SelectFromModel with Lasso (L1 Regularization), Recursive Feature Elimination (RFE) with Random Forest Regressor and
Feature selection using Variance Threshold.

Sorted feature importance using random forest and plotted.

![image](https://github.com/user-attachments/assets/db5d4458-bc2c-47b7-98dc-f91cd689682b)

Most significant variables affecting car price are highwaympg, horsepower, carlength, fuelsystem, enginesize, carwidth and so on.

# Hyperparameter Tuning

Performed hyperparameter tuning using gridsearch cv.

**After all the treatments, random forest regressor performed best with highest R2 score and lowest RMSE, MSE, and MAE.**
