# =============================================================================
# Project Name: BayesGuard
# =============================================================================
from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

app = Flask(__name__)

#HTML in Python:
BayesUI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BayesGuard UI</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body, html {
            height: 100%;
            margin: 0;
            background-color: #f7f7f7; /* Lightish grey. Easier on the eyes */
            font-family: 'Roboto', sans-serif;
        }
        .container {
            width: 50%;
            margin: auto;
            padding-top: 50px;
        }
        h1 {
            text-align: center;
            color: purple
        }
        form {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        input[type=text] {
            width: calc(100% - 25px);
            padding: 10px;
            margin-top: 5px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background-color: purple;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }
        
        #result {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>BayesGuard UI</h1>
        <form action="/process" method="post">
            <label for="filePath">Enter CSV File Path:</label>
            <input type="text" id="filePath" name="filePath" required>
            <button type="submit">Train Model</button>
        </form>

        <div id="result">{{ result|safe }}</div>
    </div>
</body>
</html>

"""

@app.route('/')
def index():
    return render_template_string(BayesUI, result="")

@app.route('/process', methods=['POST'])
def process():
    file_path = request.form['filePath']
    
    # BayesGuard python code:
    try:
        # Reading the dataset
        data = pd.read_csv("emails.csv")
        data.drop(columns=['Email No.'], inplace=True)

        x = data.iloc[:, 0:3000]
        y = data.iloc[:, 3000]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Gaussian Naive Bayes
        model = GaussianNB() 
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)

        # Counting spam and non-spam emails
        spam_count = np.sum(y_prediction == 1) 
        legit_count = np.sum(y_prediction == 0) 
        training_size = spam_count + legit_count

        # Calculating accuracy
        accuracy = np.round(accuracy_score(y_test, y_prediction), 4) * 100
        cross_val = np.round(cross_val_score(model, x, y, cv=5, scoring="accuracy").mean(), 4) * 100

        result = f"""
<div class="output">
    <strong>BayesGuard Results:</strong><br>
    <br> <!-- Inserted line break here -->
    - Accuracy: {accuracy:.2f}%<br>
    - Cross-Validation Score: {cross_val}%<br>
    <br>
    <strong>Training Details:</strong><br>
    - Training Size: {training_size}<br>
    - Spam Emails: {spam_count}<br>
    - Legitimate Emails: {legit_count}<br>
</div>
"""

    except Exception as e:
        result = f"Something went wrong: {e}"

    return render_template_string(BayesUI, result=result)

if __name__ == '__main__':
    app.run(debug=True)

# =============================================================================