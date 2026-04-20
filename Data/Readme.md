Use this for your README under a section called `Data` or `Data Access Instructions`:

````markdown
## Data Access Instructions

This project uses the IBM HR Analytics Employee Attrition dataset from Kaggle:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset is not stored directly in this repository. To reproduce the analysis, download it using `kagglehub`.

### Option 1: Download with Python
First install the package:

```bash
pip install kagglehub pandas
````

Then run:

```python
import kagglehub
import pandas as pd
from pathlib import Path

path = Path(kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset"))
print("Downloaded to:", path)

df = pd.read_csv(path / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.head())
```

### Option 2: Use the helper script

If `download_data.py` is included in this repo, run:

```bash
python download_data.py
```

This will download the dataset and save a local copy to the `data/` folder.

### Expected file

The analysis uses:

`WA_Fn-UseC_-HR-Employee-Attrition.csv`

````

And create this `download_data.py` file:

```python
from pathlib import Path
import pandas as pd
import kagglehub

DATASET = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    download_path = Path(kagglehub.dataset_download(DATASET))
    source_file = download_path / FILENAME
    target_file = data_dir / FILENAME

    df = pd.read_csv(source_file)
    df.to_csv(target_file, index=False)

    print(f"Dataset downloaded successfully.")
    print(f"Saved to: {target_file}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()
````

Your repo can then look like this:

```text
your-project/
├── Final_Project_Dat_Do.ipynb
├── README.md
├── download_data.py
└── data/
```

If you want the shortest possible version for the checkpoint, paste this in the README:

````markdown
## Data

This project uses the IBM HR Analytics Employee Attrition dataset from Kaggle:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

To download:

```python
import kagglehub
from pathlib import Path
import pandas as pd

path = Path(kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset"))
df = pd.read_csv(path / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
````

```

That should satisfy the “data/ folder or download script — data access instructions” requirement.
```
