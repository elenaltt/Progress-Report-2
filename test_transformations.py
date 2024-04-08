import pandas as pd
from etl import transform_data, data_new_variables, extract_text_features, extract_user_behavior_features  # Adjust the import according to your file structure

def test_transform_data():
    # Create a sample DataFrame
    data = {'Posting time': ['2020-01-01 12:00:00', '2021-02-02 08:30:00'],
            'Date of creation': ['2019-01-01', '2020-01-01'],
            'Source': ['Twitter Web App', 'Android'],
            'Language': ['en', 'fr'],
            'message': ['Hello world', 'Bonjour le monde']}
    df = pd.DataFrame(data)
    
    # Apply the transformation
    transformed_df = transform_data(df)
    
    # Check if unnecessary columns are removed
    assert 'Source' not in transformed_df.columns
    assert 'message' not in transformed_df.columns
    
    # Check if the filter for 'Language' == 'en' worked
    assert len(transformed_df) == 1
    assert transformed_df.iloc[0]['Language'] == 'en'

def test_data_new_variables():
    # Assume df_filtered is a DataFrame resulted from transform_data function
    data = {'Likes': [10, 5],
            'Comments/Replies': [1, 2],
            'Retweets/Reposts': [2, 1],
            'Follower': [100, 200],
            'Following': [50, 100]}
    df_filtered = pd.DataFrame(data)
    
    # Apply the transformation
    new_vars_df = data_new_variables(df_filtered)
    
    # Check if new variables are added
    assert 'ratio1' in new_vars_df.columns
    assert 'ratio2' in new_vars_df.columns
    
    # Check the correctness of the new variables
    # Note: The exact values should be based on the expected calculation from your functions
    assert abs(new_vars_df.iloc[0]['ratio1'] - 0.0833333333) < 1e-8
    assert abs(new_vars_df.iloc[0]['ratio2'] - 2.0) < 1e-8  # This is a simplified check; adjust according to your function's logic
    #assert new_vars_df.iloc[0]['ratio2'] == 2.0  # Similarly, adjust the expected value

def test_extract_text_features():
    # Creating a minimal DataFrame with necessary columns for testing
    data = {
        'Tweet Text': ['Hello world! @user #amazing http://example.com', 'Good night! ??'],
    }
    df = pd.DataFrame(data)
    
    # Apply the function to transform the data
    df_transformed = extract_text_features(df)
    
    # Assertions to verify that new columns are correctly calculated
    assert 'text_length_chars' in df_transformed.columns
    assert 'text_length_words' in df_transformed.columns
    assert 'special_char_count' in df_transformed.columns
    assert 'mentions_count' in df_transformed.columns
    assert 'urls_count' in df_transformed.columns
    assert 'sentiment_polarity' in df_transformed.columns
    assert 'capital_words_count' in df_transformed.columns
    
    # Check specific values to ensure calculations are performed correctly
    assert df_transformed.iloc[0]['text_length_chars'] == len('Hello world! @user #amazing http://example.com')
    assert df_transformed.iloc[0]['text_length_words'] == 5
    assert df_transformed.iloc[0]['special_char_count'] == 1  # "!"
    assert df_transformed.iloc[0]['mentions_count'] == 1  # "@user"
    assert df_transformed.iloc[0]['urls_count'] == 1  # "http://example.com"
    # Check for specific values in sentiment_polarity and capital_words_count based on your logic


def test_extract_user_behavior_features():
    # Creating a minimal DataFrame with necessary columns for testing
    data = {
        'Creation Year': [2019, 2021],
        'Posting Hour': [15, 3],  # 15 for Day, 3 for Night
        'Description/Self-intro': ['I love coding. #developer', 'Good vibes only. #positive'],
    }
    df = pd.DataFrame(data)
    
    # Apply the function to transform the data
    df_transformed = extract_user_behavior_features(df)
    
    # Assertions to verify that new columns are correctly calculated
    assert 'user_newness' in df_transformed.columns
    assert 'activity_mode' in df_transformed.columns
    assert 'profile_sentiment' in df_transformed.columns
    
    # Check specific values to ensure calculations are performed correctly
    assert df_transformed.iloc[0]['user_newness'] == "New"  # Assuming the reference year is 2021 and N=5
    assert df_transformed.iloc[0]['activity_mode'] == "Day"
    # The profile_sentiment will depend on the TextBlob sentiment analysis, ensure the logic matches expectations
