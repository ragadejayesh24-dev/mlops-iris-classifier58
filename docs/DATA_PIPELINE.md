\# Iris Data Pipeline



\## Overview



This project uses DVC to create a reproducible data pipeline for the Iris dataset.



The pipeline has four stages:



```text

collect → preprocess → features → validate

```



\## 1. Data Collection



\*\*Purpose:\*\* Collect the Iris dataset.



\*\*Input:\*\* Iris dataset from scikit-learn.



\*\*Output:\*\*



```text

data/raw/iris\_raw.csv

```



\## 2. Data Preprocessing



\*\*Purpose:\*\* Clean the collected data.



\*\*Input:\*\*



```text

data/raw/iris\_raw.csv

```



\*\*Processing:\*\*



\* Remove duplicate rows.

\* Clean the dataset.



\*\*Output:\*\*



```text

data/processed/iris\_preprocessed.csv

```



\## 3. Feature Engineering



\*\*Purpose:\*\* Create additional features from the preprocessed data.



\*\*Input:\*\*



```text

data/processed/iris\_preprocessed.csv

```



\*\*Features created:\*\*



\* `sepal\_area`

\* `petal\_area`

\* `sepal\_to\_petal\_length\_ratio`

\* `petal\_length\_bin`



\*\*Output:\*\*



```text

data/processed/iris\_features.csv

```



\## 4. Data Validation



\*\*Purpose:\*\* Check that the final dataset satisfies the required data-quality rules.



\*\*Input:\*\*



```text

data/processed/iris\_features.csv

```



\*\*Validation checks:\*\*



\* Required columns are present.

\* Missing values are checked.

\* Species values are valid.

\* Measurement values are within the expected ranges.



A successful validation produces:



```text

Validation PASSED

```



\## DVC Pipeline



The complete pipeline is:



```text

collect

&#x20;  ↓

preprocess

&#x20;  ↓

features

&#x20;  ↓

validate

```



The pipeline can be displayed using:



```bash

dvc dag

```



The complete pipeline can be executed using:



```bash

dvc repro

```



If nothing has changed, DVC skips the stages and reports:



```text

Data and pipelines are up to date.

```



\## Conclusion



The Iris data pipeline successfully performs data collection, preprocessing, feature engineering, and validation using DVC. The pipeline is reproducible and DVC manages its dependencies and outputs.



