import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib

def tune_isolation_forest(train_path, output_model_path):
    # Step 1: load and preprocess data
    df = pd.read_csv(train_path)
    df['labels'] = (
        df['labels']
          .apply(lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x)
          .str.rstrip('.')
    )

    # Step 2: build feature matrix and binary labels
    X = df.drop(columns=['labels']).astype('float32')
    y = (df['labels'] != 'normal').astype(int)

    print("\nClass distribution:")
    print(y.value_counts())

    # Step 3: define hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_samples': [128, 256, 'auto'],
        'max_features': [0.5, 0.7, 1.0],
        'contamination': [0.01, 0.05, 0.1],
        'bootstrap': [True, False]
    }

    iso = IsolationForest(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=iso,
        param_distributions=param_grid,
        n_iter=20,
        scoring='accuracy',
        cv=cv,
        n_jobs=1,
        verbose=2,
        refit=True,
        error_score='raise',
        random_state=42
    )

    print("\nStarting hyperparameter search (accuracy)...")
    search.fit(X, y)

    print("\nBest params (by accuracy):", search.best_params_)
    joblib.dump(search.best_estimator_, output_model_path)
    return search.best_estimator_

if __name__ == "__main__":
    TRAIN = r"C:\Users\huyng\OneDrive\Documents\SDSU\CS 549\kddcup99_10_percent_train_resampled.csv"
    MODEL_OUT = 'iforest_tuned.pkl'
    best_if = tune_isolation_forest(TRAIN, MODEL_OUT)
    print(f"\nTuned IsolationForest saved to {MODEL_OUT}")