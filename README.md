# Progress-Report-2
Progress Report 2 for WUSTL Data Wrangling

This project focuses on the Extract, Transform, Load (ETL) process specifically tailored for processing Twitter data. 
The aim is to prepare the data for further analysis （fake twitter detection） or machine learning tasks by cleaning, transforming, and enriching it with additional features. 

Function Descriptions:

ETL Functions： 
read_csv_file(file_path): 

step1: Reads a CSV file into a Pandas DataFrame, handling malformed files by skipping bad lines.

transform_data(df):

step2: Converts date-time columns to datetime objects. Extracts components like year, month, day, and hour from date-time columns.
step3: Filters rows based on the language, keeping only English tweets.
step4: Deletes unnecessary columns.

data_new_variables(df_filtered):

step5: Adds new numerical variables such as ratio1 and ratio2, calculated from existing columns like likes, comments/replies, and retweets/reposts.

extract_text_features(df):

step6: Enriches the dataset with text analysis features including text length, special character count, mentions count, sentiment polarity, and capital words count.

extract_user_behavior_features(df):

step7： Adds user behavior features such as user newness and activity mode based on the creation year and posting hour. It also includes sentiment analysis of the user's self-introduction.

Testing with Pytest
The test_transformations.py file contains unit tests for verifying the correctness of each ETL function. 
These tests use the pytest framework to ensure the data is processed as expected.
