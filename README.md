# Apply KNN Classification on Titanic Survivors Dataset
This project applies k-nearest neighbours classification to data on titanic survivors and victims. The target variable is whether they survived (=1 is survived, and =0 otherwise).

# Dataset
Original dataset was obtained from Kaggle, https://www.kaggle.com/c/titanic/data. The original dataset contains demographic information on survivors and victims as well as passenger information such as ticket fare, number of family members on board etc. 
The dataset was cleaned to encode categorical variables, impute missing values and perform logarithmic transformation to some variables. The cleaned dataset is found in cleaned_titanic.csv

# Files
* data_cleaning.py: imputes, normalises and one-hot encodes relevant features  
* classify_knn.py: fits KNN model to data

# Disclaimer
The code supplied in the zip file, fomlads.zip, is obtained for the course material of INST0060 (Foundations of Machine Learning and Data Science), a module offered by University College London, Department of Information Studies. 
