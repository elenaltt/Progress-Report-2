
import pandas as pd
import re
from textblob import TextBlob

#step1: load data
def read_csv_file(file_path):
    """
    Function to read a CSV file and return a pandas DataFrame.
    It attempts to handle malformed files by skipping bad lines.
    """
    try:
        # Read the CSV file using Python engine and skip bad lines
        data = pd.read_csv(file_path, error_bad_lines=False, engine='python')
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

#step2: Time variables transformation  
#step3: deletes unnecessary columns including the 'Source', and 'message'  
#step4: filter the data with language == english

def transform_data(df):
    """
    Transforms the data by converting datetime columns to datetime objects,
    extracting components, deleting unnecessary columns, and filtering rows.
    """
    # Convert 'Posting time' and 'Date of creation' to datetime
    df['Posting time'] = pd.to_datetime(df['Posting time'], errors='coerce')
    df['Date of creation'] = pd.to_datetime(df['Date of creation'], errors='coerce')

    # Check if conversion was successful and process datetime columns
    if df['Posting time'].dtype == '<M8[ns]' and df['Date of creation'].dtype == '<M8[ns]':
        df['Posting Year'] = df['Posting time'].dt.year
        df['Posting Month'] = df['Posting time'].dt.month
        df['Posting Day'] = df['Posting time'].dt.day
        df['Posting Hour'] = df['Posting time'].dt.hour
        
        df['Creation Year'] = df['Date of creation'].dt.year
        df['Creation Month'] = df['Date of creation'].dt.month
        df['Creation Day'] = df['Date of creation'].dt.day
    else:
        print("Error: Datetime conversion failed for one or more columns.")

    # Delete the original datetime columns and other unnecessary columns
    df.drop(['Source', 'message'], axis=1, inplace=True)
    
    # Filter rows where 'Language' is 'en'
    df_filtered = df[df['Language'] == 'en']

    return df_filtered

# step5: transform data with adding new numerical variables
def data_new_variables(df_filtered):

    # make sure 'Likes', 'Comments/Replies', and 'Retweets/Reposts' are numerical
    df_filtered['Likes'] = pd.to_numeric(df_filtered['Likes'], errors='coerce').fillna(0)
    df_filtered['Comments/Replies'] = pd.to_numeric(df_filtered['Comments/Replies'], errors='coerce').fillna(0)
    df_filtered['Retweets/Reposts'] = pd.to_numeric(df_filtered['Retweets/Reposts'], errors='coerce').fillna(0)

    # 计算ratio1
    df_filtered['ratio1'] = df_filtered['Comments/Replies'] / (df_filtered['Retweets/Reposts'] + df_filtered['Likes'] + 1e-8)  # 分母加上一个小正数避免除以零
    df_filtered['ratio2'] = df_filtered['Follower'] / (df_filtered['Following'] + 1e-8)

    
    return df_filtered

# step6: transform data with adding new text analysis variables (text analysis on twitter text)

def extract_text_features(df):
    # Text length (in characters)
    df['text_length_chars'] = df['Tweet Text'].apply(len)
    
    #  Text length (in words)
    df['text_length_words'] = df['Tweet Text'].apply(lambda x: len(x.split()))
    
    # Number of special characters
    df['special_char_count'] = df['Tweet Text'].apply(lambda x: len(re.findall(r'[\!\?]', x)))
    
    # Number of references to users
    df['mentions_count'] = df['Tweet Text'].apply(lambda x: len(re.findall(r'@\w+', x)))
    
    # Number of links included
    df['urls_count'] = df['Tweet Text'].apply(lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)))
    
    # Sentiment analysis (emotional polarity)
    df['sentiment_polarity'] = df['Tweet Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Number of words with initial capital letters
    df['capital_words_count'] = df['Tweet Text'].apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', x)))
    
    return df
#step7: extract user behavior features 

def extract_user_behavior_features(df):
    """
    Extracts user behavior features including user newness and activity mode.
    extract sentiment of user from the 
    
    Parameters:
    - df: A pandas DataFrame with 'Creation Year' and 'Posting Hour' and 'Description/Self-intro' columns.
    
    Returns:
    - A pandas DataFrame with new features 'user_newness' and 'activity_mode' and 'profile_sentiment'.
    """
    # classify users as new users or old users
    reference_year = df['Creation Year'].max()  # set the latest year as reference
    N = 5  # Define "new user" as the last N years. 
    df['user_newness'] = df['Creation Year'].apply(lambda x: "New" if (reference_year - x) <= N else "Old")

    # User's activity mode variable
    def classify_activity_hour(hour):
        if 0 <= hour <= 6 or 18 <= hour <= 23:
            return "Night"
        else:
            return "Day"

    df['activity_mode'] = df['Posting Hour'].apply(classify_activity_hour)

    df['profile_sentiment'] = df['Description/Self-intro'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    return df




def main():
    file_path = '90new.csv'  
    df = read_csv_file(file_path)

    if df is not None:
        # Display the current length of the DataFrame
        print(f"Current length of the DataFrame: {len(df)}")

        # Display the column names in the DataFrame
        print("Current columns in the DataFrame:")
        print(list(df.columns))

        # Optionally, display the first few rows of the DataFrame
        print("Displaying the first few rows of the DataFrame:")
        print(df.head())

        
        df_filtered = transform_data(df)
        
        # Display the number of rows after filtering
        print(f"Number of rows after filtering: {len(df_filtered)}")

        # Display the first few rows of the filtered DataFrame
        print("Displaying the first few rows of the filtered DataFrame:")
        print(df_filtered.head())

        df_new_variable = data_new_variables(df_filtered)
        print("Current variables in the df_new_variable:")
        print(list(df_new_variable.columns))

        df_text_analysis = extract_text_features(df_new_variable)
        print("Current variables in the df_text_analysis:")
        print(list(df_text_analysis.columns))

        df_user_analysis = extract_user_behavior_features(df_text_analysis)
        print("Current variables in the :df_user_analysis")
        print(list(df_user_analysis.columns))

        df_user_analysis.to_csv('90news_processed', index=False)








if __name__ == "__main__":
    main()
