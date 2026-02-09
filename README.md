# CS549 Final Project: Anomaly Detection in Network Traffic

This repository contains Group 4’s final project for CS549, focused on detecting anomalies in the KDD Cup 1999 dataset using a mix of supervised and unsupervised machine learning models.

## Team Members

- Trevor Thayer — Random Forests, Repository Structure, Documentation
- Anthony Do — SVM, Preprocessing, Documentation  
- Anh Huy Nguyen — Isolation Forests, Feature Scaling, Preprocessing, Documentation
- Isabelle Viraldo — K-Means Clustering, Project Objectives, Documentation  

## Repo Structure

```

CS549-FINAL/
│
├── data/                      ← Raw and processed KDD99 dataset CSVs (not included in zip file, but where the code expects them)
├── IsolationForest/           ← Code + README for Isolation Forest
├── KMeansClustering/          ← Code + README for K-Means
├── RandomForest_Trevor/       ← Code + README for Random Forest
├── SVM_Model/                 ← Code + README for SVM
├── utils/                     ← Preprocessing script (required)
├── requirements.txt           ← Required Python packages
├── README.md                  ← This file

```

## Project Setup

Before running any of the models, create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing Required

All models rely on preprocessed CSVs found in the `/data/` folder.

To generate these files, run the following script: Ensure that these csv's are in the `/data/` folder when running the models.

```bash
python3 utils/preprocess_andi.py
```

This script:

* Loads the 10% version of the KDD Cup 1999 dataset
* Cleans missing values
* One-hot encodes categorical features
* Scales numerical columns
* Handles class imbalance using SMOTE

These are required for running the models.

## How to Run the Models

Each model is modularized in its own directory with a dedicated `README.md` that contains instructions on running the code and interpreting the results.

###Refer to each folder's README for specific runtime entry points.

## Requirements

Install them all via:

```bash
pip install -r requirements.txt
```

## References

* [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
