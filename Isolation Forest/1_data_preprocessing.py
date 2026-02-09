from sklearn.datasets import fetch_kddcup99
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

kdd_data = fetch_kddcup99(return_X_y=True, as_frame=True, percent10=True)

X, y = kdd_data

df = pd.concat([X, y], axis=1)

df.to_csv('kddcup99_10_percent_data.csv', index=False)

def handle_missing_values(csv_path, output_csv=None):
    """
    Reads a CSV file, identifies missing or inconsistent data, 
    and fills missing values using:
      - Mean for numerical columns
      - Mode for categorical columns.
      
    Parameters:
      csv_path (str): Path to the CSV file.
      output_csv (str, optional): If provided, the cleaned DataFrame will be saved to this file.
      
    Returns:
      pd.DataFrame: The cleaned DataFrame.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Display missing values before imputation
    print("Missing values BEFORE imputation:")
    print(df.isnull().sum())
    
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Impute numerical columns using the mean
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    # Impute categorical columns using the mode (most frequent value)
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Display missing values after imputation
    print("\nMissing values AFTER imputation:")
    print(df.isnull().sum())
    
    # Optionally, save the cleaned DataFrame back to a CSV file
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nCleaned data saved to: {output_csv}")
    
    return df

# # Example usage:
# if __name__ == "__main__":
#     csv_file = 'kddcup99_10_percent_data.csv'
#     cleaned_df = handle_missing_values(csv_file, output_csv='kddcup99_10_percent_cleaned.csv')

def encode_categorical_features(df, categorical_columns=['protocol_type', 'service', 'flag'], output_csv=None):
    """ Encodes specified categorical features using one-hot encoding.
    
    Parameters:
      df (pd.DataFrame): The cleaned DataFrame (output from the missing values function).
      categorical_columns (list): List of categorical column names to encode.
      output_csv (str, optional): If provided, the encoded DataFrame will be saved to this file.
      
    Returns:
      pd.DataFrame: DataFrame with categorical features replaced by one-hot encoded columns."""
    # Create one-hot encoded columns for the specified categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    
    # Optionally save the encoded DataFrame to a CSV file
    if output_csv:
        df_encoded.to_csv(output_csv, index=False)
        print(f"Encoded data saved to: {output_csv}")
    
    return df_encoded

def scale_numerical_features(df, numerical_columns, method='standard', output_csv=None):
    """
    Scales numerical features in the DataFrame.
    
    Parameters:
      df (pd.DataFrame): The DataFrame containing the data.
      numerical_columns (list): List of column names that are numerical and need scaling.
      method (str): The scaling method; 'standard' uses StandardScaler, 
                    'minmax' uses MinMaxScaler. Default is 'standard'.
      output_csv (str, optional): If provided, the scaled DataFrame is saved to this CSV file.
    
    Returns:
      pd.DataFrame: The DataFrame with the scaled numerical features.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    # Fit the scaler on the numerical columns and transform them
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Optionally save the scaled DataFrame to a CSV file
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Scaled data saved to: {output_csv}")
    
    return df

def handle_imbalanced_data(X, y, method='smote', random_state=42, k_neighbors=1):
    """
    Handles class imbalance by applying oversampling or undersampling techniques.
    
    Parameters:
      X (pd.DataFrame or np.array): Feature matrix from the training set.
      y (pd.Series or np.array): Target vector from the training set.
      method (str): The method to address imbalance. Options are:
                    - 'smote' (default): Synthetic Minority Over-sampling Technique.
                    - 'random_oversampling': Randomly duplicate minority class examples.
                    - 'random_undersampling': Randomly remove examples from majority classes.
      random_state (int): Random seed for reproducibility.
      k_neighbors (int): Number of nearest neighbors for SMOTE (only applicable if method=='smote').
    
    Returns:
      X_res (array-like): Resampled feature matrix.
    """
    if method.lower() == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    elif method.lower() == 'random_oversampling':
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=random_state)
    elif method.lower() == 'random_undersampling':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError("Unsupported method. Choose 'smote', 'random_oversampling', or 'random_undersampling'.")
    
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

def main():
    # Step 1: Load and save raw data
    print("\n" + "="*40)
    print("Step 1/5: Loading raw data...")
    kdd_data = fetch_kddcup99(return_X_y=True, as_frame=True, percent10=True)
    X, y = kdd_data

        # decode
    if isinstance(y.iloc[0], bytes):
        y = y.map(lambda val: val.decode('utf-8'))

    df = pd.concat([X, y], axis=1)
    df.to_csv('kddcup99_10_percent_data.csv', index=False)
    print("Raw data saved to: kddcup99_10_percent_data.csv")

    # Step 2: Handle missing values
    print("\n" + "="*40)
    print("Step 2/5: Handling missing values...")
    df_clean = handle_missing_values(
        'kddcup99_10_percent_data.csv',
        output_csv='kddcup99_10_percent_cleaned.csv'
    )

    # Step 3: Encode categorical features
    print("\n" + "="*40)
    print("Step 3/5: Encoding categorical features...")
    df_encoded = encode_categorical_features(
        df_clean,
        categorical_columns=['protocol_type', 'service', 'flag'],
        output_csv='kddcup99_10_percent_encoded.csv'
    )

    # Step 4: Scale numerical features
    print("\n" + "="*40)
    print("Step 4/5: Scaling numerical features...")
    # Identify numerical columns (exclude categorical and target)
    numerical_columns = [col for col in df_encoded.columns 
                       if col not in ['protocol_type', 'service', 'flag', 'labels']]
    
    df_scaled = scale_numerical_features(
        df_encoded,
        numerical_columns=numerical_columns,
        method='standard',
        output_csv='kddcup99_10_percent_scaled.csv'
    )

    # Step 5: Handle class imbalance
    print("\n" + "="*40)
    print("Step 5/5: Handling class imbalance...")
    X = df_scaled.drop(columns=['labels'])
    y = df_scaled['labels']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Resample training data
    X_train_res, y_train_res = handle_imbalanced_data(
        X_train, y_train, method='smote', k_neighbors=1
    )
    
    # Save final datasets
    pd.concat([X_train_res, y_train_res], axis=1).to_csv(
        'kddcup99_10_percent_train_resampled.csv', index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        'kddcup99_10_percent_test.csv', index=False
    )
    
    print("\nPreprocessing complete!")
    print("Final training data: kddcup99_10_percent_train_resampled.csv")
    print("Test data: kddcup99_10_percent_test.csv")

if __name__ == "__main__":
    main()

        



