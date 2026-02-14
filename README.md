# Study Repository

This repository contains my structured daily learning progress in **Python** and **AI & ML**.  
Each folder is organized day-wise with concepts covered, short theory explanations, and practical implementations.



# Python Folder



## Day 1 – Python Basics & Data Structures

### Theory

**Python Basics:**  
Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

**Variables & Data Types:**  
Variables are used to store data values. Python supports multiple data types like integers, floats, strings, and booleans. Type casting allows conversion between different data types.

**Conditional Statements:**  
Conditional statements (`if`, `elif`, `else`) are used to make decisions in programs based on certain conditions.

**Lists:**  
Lists are ordered, mutable collections that can store multiple values. They support indexing, slicing, and built-in methods like append(), pop(), remove(), etc.

### Activities Performed
- Practiced variables and type casting
- Built voting eligibility checker
- Created even/odd checker
- Implemented salary decision program
- Practiced list indexing and slicing
- Modified list elements using list methods


## Day 2 – Tuple, Set, Dictionary & Loops

### Theory

**Tuples:**  
Tuples are ordered and immutable collections. Once created, their elements cannot be modified.

**Sets:**  
Sets are unordered collections that store unique values. They are mainly used to remove duplicates and perform mathematical set operations.

**Dictionaries:**  
Dictionaries store data in key-value pairs. They allow fast data retrieval using keys.

**Loops:**  
Loops (`for`, `while`) are used to repeat a block of code multiple times. Nested loops are loops inside another loop.

### Activities Performed
- Removed elements from tuple using slicing
- Performed set operations (add, remove, update)
- Created dictionary with nested values
- Counted character frequency in string
- Printed patterns using nested loops
- Separated even and odd numbers from list



## Day 3 – While Loop, Functions & OOP

### Theory

**While Loop:**  
The while loop executes a block of code as long as a condition remains true. It is useful when the number of iterations is unknown.

**Functions:**  
Functions are reusable blocks of code that perform a specific task. They improve modularity and code readability.

**Object-Oriented Programming (OOP):**  
OOP is a programming paradigm based on classes and objects. It allows encapsulation, reusability, and better structure of code.

### Activities Performed
- Printed numbers using while loop
- Built interactive calculator using functions
- Created Rectangle class
- Implemented ATM system using class and methods
- Practiced object creation and method calling


## Day 4 – NumPy & Statistics

### Theory

**NumPy:**  
NumPy is a powerful numerical computing library in Python. It provides support for arrays and mathematical operations on large datasets.

**Statistical Measures:**  
Mean, median, and mode are measures of central tendency. Variance and standard deviation measure data dispersion.

**Quantiles:**  
Quantiles divide data into equal-sized intervals and help understand data distribution.

### Activities Performed
- Created NumPy arrays
- Performed element-wise operations
- Calculated mean, median, and mode
- Computed variance and standard deviation
- Used quantile functions for data analysis



## Day 5 – Pandas & Data Analysis

### Theory

**Pandas:**  
Pandas is a Python library used for data manipulation and analysis. It provides DataFrame and Series structures to handle structured data efficiently.

**Data Cleaning & EDA:**  
Exploratory Data Analysis (EDA) helps understand datasets through summary statistics, visualization, and data cleaning.

**GroupBy & Aggregation:**  
GroupBy allows grouping data based on categories and performing aggregate functions like sum, mean, count, etc.

### Activities Performed
- Loaded CSV files
- Created and dropped columns
- Used iloc for slicing
- Concatenated DataFrames
- Performed GroupBy on IPL dataset
- Conducted EDA on Employees dataset
- Checked null values using info() and describe()



# AI & ML Folder

This folder contains my practical implementation of Machine Learning algorithms including Regression, Classification, Clustering, and Neural Networks. Each day includes theory understanding along with real dataset implementation.



## ML_Day_1 – Simple Linear Regression

### Theory

**Linear Regression:**  
Linear Regression is a supervised learning algorithm used to predict continuous values. It models the relationship between an independent variable (X) and a dependent variable (y) using a straight line equation.

**Model Evaluation Metrics:**  
Mean Squared Error (MSE) measures prediction error, and R² Score indicates how well the model explains the variance in the data.

### Implementation Details

- Dataset: `appliance_energy.csv`
- Feature: Temperature (°C)
- Target: Energy Consumption (kWh)
- Split data using `train_test_split()`
- Trained model using `LinearRegression()`
- Evaluated using:
  - **MSE:** 0.1634
  - **R² Score:** 0.6119
- Visualized regression line using Matplotlib
- Saved trained model using `joblib` → `appliance_energy_model.pkl`



## ML_Day_2 – Clustering (K-Means) & Logistic Regression  
*(Both algorithms implemented on the same day)*



### Part 1 – K-Means Clustering

### Theory

**K-Means Clustering:**  
K-Means is an unsupervised learning algorithm used to group similar data points into clusters. It minimizes inertia (within-cluster sum of squares) to form compact clusters.

**Elbow Method:**  
The Elbow Method helps determine the optimal number of clusters by plotting inertia against different K values.

### Implementation Details

- Dataset: `Mall_Customers.csv`
- Selected features:
  - Annual Income (k$)
  - Spending Score (1-100)
- Applied `KMeans()` with multiple K values
- Used Elbow Method (K = 1 to 30)
- Final model selected with **5 clusters**
- Extracted:
  - Cluster labels
  - Cluster centroids
- Visualized clusters using Seaborn scatterplot
- Compared inertia values to evaluate clustering quality



### Part 2 – Logistic Regression

### Theory

**Logistic Regression:**  
Logistic Regression is a supervised classification algorithm used for binary classification problems. It predicts probability values between 0 and 1 using the sigmoid function.

**Evaluation Metrics:**  
Accuracy, Confusion Matrix, Precision, Recall, and F1-score are used to measure classification performance.

### Implementation Details

- Dataset: `green_tech_data.csv`
- Features:
  - carbon_emissions
  - energy_output
  - renewability_index
  - cost_efficiency
- Target:
  - sustainability (0 or 1)

- Split data into training & testing sets
- Trained model using `LogisticRegression()`
- Model Accuracy: **95%**
- Generated:
  - Confusion Matrix (Heatmap)
  - Classification Report
- Saved trained model using `joblib` → `lrmodel_sustainable.pkl`



## ML_Day_3 – FNN (Feed Forward Neural Network)

### Theory

**Feed Forward Neural Network (FNN):**  
A Feed Forward Neural Network is a type of Artificial Neural Network where data flows in one direction — from input layer to hidden layers to output layer.

**Deep Learning for Regression:**  
Neural networks can model complex non-linear relationships and are trained using backpropagation and optimization algorithms like Adam.

### Implementation Details

- Dataset: `predict_energy_consumption.csv`
- Features:
  - temperature
  - humidity
  - wind_speed
  - solar_irradiance
- Target:
  - energy_consumption

- Performed:
  - Train-test split
  - Feature scaling using `StandardScaler`

- Built Neural Network:
  - Dense Layer (64 neurons, ReLU)
  - Dense Layer (32 neurons, ReLU)
  - Output Layer (1 neuron for regression)

- Compiled model with:
  - Optimizer: Adam
  - Loss: Mean Squared Error
  - Metric: Mean Absolute Error

- Trained for 50 epochs
- Test Mean Absolute Error: **~115**
- Visualized:
  - Training vs Validation Loss
  - True vs Predicted values scatter plot



# Libraries Used in AI & ML

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow / keras
- joblib
