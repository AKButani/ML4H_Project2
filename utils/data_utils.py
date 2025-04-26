def remove_label(df):
    # Extract the 'HeartDisease' column as the labels
    labels = df['HeartDisease']
    
    # Remove the 'HeartDisease' column from the dataframe
    df_without_heart_disease = df.drop(columns=['HeartDisease'])
    
    # Return the modified DataFrame and the labels
    return df_without_heart_disease, labels

def process_cat_vars(df):
    """
    Function to preprocess the data, including encoding categorical variables and 
    applying one-hot encoding to certain columns. This function is designed to be applied
    to both the training and test sets.
    
    Parameters:
    df (DataFrame): The input dataframe containing the features and the target label.

    Returns:
    DataFrame: The preprocessed dataframe
    """
    # Encode categorical variables
    # Male is 1, female is 0 for 'Sex'
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    
    # 'ExerciseAngina' : Yes is 1 and No is 0
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    
    # 'ST_Slope': Up is 1, Flat is 0, Down is -1
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': -1})

    # One hot encoding for categorical columns that have more than two categories
    df_encoded = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'], dtype=int)
    
    return df_encoded
