## Data Access Instructions

This project uses the IBM HR Analytics Employee Attrition dataset from Kaggle:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset is not stored directly in this repository. To reproduce the analysis, download it using `kagglehub`.

### Option 1: Download with Python
First install the package:

```bash
!pip install kagglehub pandas
import kagglehub
import pandas as pd
from pathlib import Path

path = Path(kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset"))
print("Downloaded to:", path)

df = pd.read_csv(path / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.head())
