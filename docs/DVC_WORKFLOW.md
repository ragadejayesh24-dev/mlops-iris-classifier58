# DVC Workflow

## 1. DVC Remote Configuration

A local DVC remote was configured for storing dataset versions.

Remote location:

C:/Users/A S U S/dvc-storage

The remote was added using:

dvc remote add -d local ~/dvc-storage

## 2. DVC Workflow for Dataset Changes

The workflow used for each dataset change was:

1. Modify or generate the dataset.
2. Track the dataset using:

   dvc add data/raw/iris_v1.csv

3. Stage the DVC metadata:

   git add data/raw/iris_v1.csv.dvc

4. Commit the metadata to Git:

   git commit -m "commit message"

5. Push the dataset to the DVC remote:

   dvc push

Git stores the DVC metadata while DVC stores the actual dataset version.

## 3. Comparing Dataset Versions

The dataset versions were compared using:

dvc diff a71c232

This showed that the Iris dataset was modified between Version 1 and Version 2.

## 4. Restoring Dataset Versions

A previous dataset version can be restored by checking out its `.dvc` file and running:

dvc checkout data/raw/iris_v1.csv.dvc

Version 1 contained 150 data rows.

Version 2 contained 170 data rows.

The dataset was successfully restored between these two versions during the experiment.

## 5. Git and DVC Roles

Git tracks:

- DVC metadata files
- Source code
- Documentation

DVC tracks:

- Large dataset files
- Dataset versions
- Dataset storage

This allows the project to maintain reproducible versions of the machine learning dataset.
