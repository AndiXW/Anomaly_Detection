Step 1: Run the 1_data_preprocessing file
- This may take up to 1+ hours and take up to 13-14 GB of memory
- Six new .csv files will be created, the two final files that are used for steps 2 and 3 are kddcup99_10_percent_train_resampled and kddcup99_10_percent_test

Step 2: Run the 2_IsolationForest file
- This may take up to 2+ hours for the model training process
- At code line 54, change the file path to your own kddcup99_10_percent_train_resampled's file path on your computer
- New file iforest_tuned.pkl will be created

Step 3: Run the 3_Evaluation_IsolationForest file
- It only takes a few seconds to run
- At code lines 154 and 155, change the file path to your own iforest_tuned.pkl's and kddcup99_10_percent_test's file path on your computer
- Three new result graphs will be created, and metrics' stats will be printed out on the terminal 
