\# AutoJudge: Programming Problem Difficulty Prediction



\## Project Overview

AutoJudge is a machine learning-based system that predicts the difficulty of programming problems using textual descriptions. The system performs both classification (Easy / Medium / Hard) and regression (difficulty score prediction) using classical machine learning techniques.



\## Dataset Used

The dataset consists of programming problem descriptions stored in JSONL format.

Each problem includes:

\- Title

\- Problem description

\- Input description

\- Output description

\- Sample input/output

\- Difficulty class

\- Difficulty score



Source:

https://github.com/AREEG94FAHAD/TaskComplexityEval-24



\## Approach and Models Used



\### Data Preprocessing

\- Converted JSONL data into a structured dataset

\- Handled missing values

\- Flattened nested input/output structures

\- Combined all text fields into a single feature

\- Cleaned and normalized text



\### Feature Extraction

\- TF-IDF vectorization

\- Custom features such as:

&nbsp; - Text length

&nbsp; - Mathematical symbol count

\- Feature scaling using StandardScaler



\### Classification Models

\- Logistic Regression (baseline)

\- Support Vector Machine (SVM) – final model



\### Regression Models

\- Gradient Boosting Regressor – final model



Deep learning models were not used, in accordance with project guidelines.



\## Evaluation Metrics



\### Classification

\- Accuracy

\- Confusion Matrix



\### Regression

\- Mean Absolute Error (MAE)

\- Root Mean Squared Error (RMSE)



\## Web Interface

A Streamlit-based web application allows users to input problem descriptions and receive:

\- Predicted difficulty class

\- Predicted difficulty score



The application runs locally without requiring deployment.



\## Steps to Run the Project Locally



1\. Clone the repository

2\. Install required libraries:

&nbsp;  pip install -r requirements.txt

3\. Run the web application:

&nbsp;  streamlit run app.py

4\. Open the browser at:

&nbsp;  http://localhost:8501



\## Demo Video

Demo video link (2–3 minutes):

ADD YOUR VIDEO LINK HERE



\## Author

Name: Sahithi Golla

Institution: IIT Roorkee  

Domain: Data Science / Machine Learning



