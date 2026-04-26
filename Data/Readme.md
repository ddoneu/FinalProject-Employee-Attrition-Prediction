## Data Access Instructions

The dataset is included in this repository at `Data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

**Source:** IBM HR Analytics Employee Attrition Dataset  
**Kaggle URL:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  
**Access date:** April 2026

### Option 1: Use the file already in this repo (recommended)

The CSV is already in the `Data/` folder. The notebook and Streamlit app load it directly from the raw GitHub URL — no download or API key required.

```python
import pandas as pd

url = "https://raw.githubusercontent.com/ddoneu/FinalProject-Employee-Attrition-Prediction/refs/heads/main/Data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(url)
print(df.shape)
```

### Option 2: Download fresh from Kaggle

Install kagglehub first:

```bash
pip install kagglehub
```

Then run:

```python
import kagglehub
import pandas as pd
from pathlib import Path

path = Path(kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset"))
df = pd.read_csv(path / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.shape)
```

### Expected file

```
WA_Fn-UseC_-HR-Employee-Attrition.csv
Shape: (1470, 35)
```
