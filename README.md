Rain Prediction Project – Detailed README

Project Title
P13 – Will It Rain Tomorrow? (Binary Weather Classification using Machine Learning)

Team Members
24bec010@iiitdwd.ac.in
24bec015@iiitdwd.ac.in
24bec023@iiitdwd.ac.in
24bec062@iiitdwd.ac.in

1. Introduction

This project focuses on predicting whether it will rain the next day based on historical weather observations. Weather prediction is an important application of data analysis and machine learning because accurate rainfall forecasts can help in multiple areas including agriculture, transportation, outdoor planning, and flood prevention.

In this project, we use the publicly available UK Weather Observations dataset from the London Datastore. We implement the entire workflow inside a single Jupyter Notebook named project.ipynb. The notebook contains all the stages of a machine learning pipeline: data loading, exploration, cleaning, preprocessing, feature extraction, model selection, training, evaluation, and final prediction.

The aim of the project is to design a simple yet effective binary classifier that takes as input the current day’s weather features and outputs whether it will rain tomorrow (Yes or No).

2. Dataset Description

The dataset used in this project is obtained from the London Datastore. It contains daily meteorological observations—for example:

• Temperature readings
• Humidity levels
• Wind speed
• Pressure values
• Rainfall amounts
• Sunshine duration
• Cloud cover
• Other weather-related features

These values are collected for multiple dates spanning several years. The dataset also includes the actual rainfall for the following day, which allows us to train a binary classifier.

The target variable created for this project is:

• 1 → It will rain tomorrow
• 0 → It will not rain tomorrow

3. Project Structure

This repository contains only one file, project.ipynb. All the steps, logic, experiments, and results are inside this single notebook. There are no additional Python scripts or modules at this stage.

The notebook is structured into clearly labeled sections to allow the user to follow the entire machine learning workflow from start to finish.

4. Workflow and Methodology
4.1 Data Loading

The dataset is imported into the notebook. Initial inspection is performed to understand the shape of the data, the number of columns, column types, missing values, and general consistency of the dataset.

4.2 Data Cleaning

Weather datasets often contain missing values, inconsistencies, formatting issues, or outliers.
In this stage, the notebook handles:

• Removal or imputation of missing values
• Conversion of string-based values into numerical form
• Correction of incorrectly formatted data
• Removal of irrelevant or redundant columns if necessary

Proper cleaning ensures that the machine learning model receives high-quality, structured data.

4.3 Feature Engineering

Feature engineering plays a crucial role in improving model performance.

The notebook performs tasks such as:

• Selecting important weather attributes
• Creating new calculated features if required
• Converting categorical features into numerical codes
• Normalizing or standardizing numerical columns
• Forming the final input feature matrix (X) and target vector (y)

The main goal at this stage is to create a set of clean, suitable features that allow the model to learn patterns related to rainfall.

4.4 Target Variable Creation

A binary target variable is created to represent whether rain will occur the following day. This transforms the problem into a classical binary classification task.

4.5 Model Selection and Training

For this project, Logistic Regression is chosen as the baseline model. Logistic Regression is a well-known algorithm for binary classification problems and performs well on structured datasets.

The notebook includes:

• Splitting data into training and testing sets
• Initializing the logistic regression classifier
• Training the model on the cleaned dataset
• Saving the trained model within the notebook’s runtime

4.6 Model Evaluation

After training, the model is evaluated on the test set.
The notebook presents:

• Accuracy score
• Confusion matrix
• Classification report (precision, recall, f1-score)
• Correct vs. incorrect predictions

These metrics help assess how well the model generalizes to unseen data.

4.7 Prediction System

The final section of the notebook performs prediction. The user can input sample weather values (or use test data), and the model outputs:

“Rain Tomorrow: Yes”
or
“Rain Tomorrow: No”

This demonstrates the practical usability of the classifier.

5. How to Use the Notebook

The project.ipynb file can be opened in any Jupyter Notebook environment. Once opened, the notebook can be executed from top to bottom. Each cell is clearly explained, and the process follows a simple linear flow:

Load data

Clean and preprocess

Train the model

Evaluate

Predict

This makes the project easy to understand even for beginners.

6. Results and Observations

The results include:

• Model accuracy on test data
• A detailed evaluation report
• Visual or textual representation of performance
• Final rainfall prediction

The trained logistic regression model generally shows decent performance on structured weather datasets. However, performance may vary depending on the range of years, missing data, and variability in weather patterns.

7. Limitations

Some limitations of the current version:

• Only one machine learning algorithm is used (Logistic Regression)
• The model may not capture complex seasonal patterns
• Dataset preprocessing is basic
• Predictions rely heavily on data quality
• Notebook-based organization may not be production-ready

These limitations are natural at this stage and can be improved in future versions.

8. Conclusion

This project successfully demonstrates the end-to-end process of building a machine learning model for rainfall prediction using real-world weather data. By following a structured workflow inside a single Jupyter Notebook, the project provides a clear understanding of how data preprocessing, logistic regression, and prediction work together to solve a binary classification problem.

The project forms a strong foundation for students to explore more advanced techniques, incorporate better models, and transform this into a fully functional predictive system.
