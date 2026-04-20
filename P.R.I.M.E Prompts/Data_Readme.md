PURPOSE

Help me write a short, clear Data/Readme.md section for my GitHub repository that explains how to access the dataset for my ECON 3916 final project. The goal is to make the repository reproducible without manually storing the dataset in the repo.

ROLE

Act as a concise technical writing and Python assistant for a student machine learning project. Write clean markdown and simple reproducible Python code. Keep the tone professional, direct, and practical.

INPUT

Repository: FinalProject-Employee-Attrition-Prediction

Project dataset:
IBM HR Analytics Employee Attrition dataset from Kaggle
Dataset URL: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset is not stored directly in the repository.
I want users to be able to reproduce the analysis by downloading the data with kagglehub.

I want the output to include:
1. A short Data Access Instructions README section
2. Option 1: Python download instructions using kagglehub
3. Option 2: mention a helper script called download_data.py
4. A Python script called download_data.py that downloads the dataset and saves the CSV into a local data/ folder
5. A short example of the expected repo structure

METHOD

Write the output in two parts:

Part A:
Create a short markdown section called “Data Access Instructions” with:
- dataset name
- Kaggle URL
- Option 1: Python download instructions using kagglehub
- Option 2: how to run download_data.py

Part B:
Write a clean Python script called download_data.py that:
- imports Path, pandas, and kagglehub
- downloads the dataset from Kaggle
- creates a local data/ folder if needed
- reads the expected CSV file
- saves it into the local data/ folder
- prints the saved file path and dataset shape

Keep everything short, clean, and reproducible.
Do not invent extra files or features.
Do not add explanations outside the requested output.

EXPECTED OUTPUT

Return exactly:
1. A markdown block for Data/Readme.md
2. A Python code block for download_data.py
3. A short text block showing the repo structure

Do not include any extra commentary outside those three outputs.
