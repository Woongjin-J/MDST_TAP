import numpy as np
import pandas as pd
import os

state_abbrs = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
    'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

def categorize_street(street_name):
    street_name = str(street_name)  # Ensure it's a string

    # Check for highways
    if any(x in street_name for x in ['I-', 'US-', 'State Route', 'SR-',
                                      'Hwy', 'Highway', 'Fwy', 'Expressway', 'Expy',
                                      'Turnpike', 'Pike', 'Pkwy', 'Interstate', 'Trwy', 'Tpke',
                                      'Bypass', 'Corridor']) or \
       any(f"{abbr}-" in street_name for abbr in state_abbrs):  # Detect state-specific highways
        return 'Highway/Freeway'

    # Check for local roads
    elif any(x in street_name for x in ['Ave', 'Rd', 'St', 'Blvd', 'Dr', 'Ct', 'Pl', 'Ln', 'Way', 'Trail', 'Plaza']):
        return 'Local Road'

    # Default category
    else:
        return 'Other'

def encode_street_type(street_type):
    if pd.isna(street_type):
        return 0  # Default to local road if missing

    street_type = str(street_type)

    # Highways/Freeways
    if street_type == 'Highway/Freeway':
        return 1  # Highway/Freeway
    else:
        return 0  # Local Road / Other

def encode_traffic_signal(signal):
    return 1 if bool(signal) else 0

def encode_crossing(crossing):
    return 1 if bool(crossing) else 0

def preprocess_time_features(df):
    df = df.copy()
    # Convert time columns to datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    # Drop rows where Start_Time or End_Time is missing
    df = df.dropna(subset=['Start_Time', 'End_Time'])

    # # Extract features
    df['Start_Hour'] = df['Start_Time'].dt.hour  # Hour of the day
    df['Start_Month'] = df['Start_Time'].dt.month  # Month of the year
    df['Accident_Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60  # Duration in minutes

    # Fill missing values with median
    df['Accident_Duration'] = df['Accident_Duration'].fillna(df['Accident_Duration'].median())

    # # Encode cyclic time features
    df['Start_Hour_Sin'] = np.sin(2 * np.pi * df['Start_Hour'] / 24)
    df['Start_Hour_Cos'] = np.cos(2 * np.pi * df['Start_Hour'] / 24)

    df['Start_Month_Sin'] = np.sin(2 * np.pi * df['Start_Month'] / 12)
    df['Start_Month_Cos'] = np.cos(2 * np.pi * df['Start_Month'] / 12)

    # Fill missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

if __name__ == "__main__":
    print("Loading US Accidents dataset...")

    file_path = 'data/US_Accidents_March23.csv'

    # Load dataset with random sampling
    us_accidents = pd.read_csv(file_path).sample(frac=0.013, random_state=42)

    # Verify dataset size
    print(f"Sample size: {len(us_accidents)}")

    print("Preprocessing the dataset...")

    # Preprocess time features
    us_accidents = preprocess_time_features(us_accidents)

    # Normalize distance to prevent scale issues
    us_accidents['Distance(mi)'] = (us_accidents['Distance(mi)'] - us_accidents['Distance(mi)'].mean()) / us_accidents['Distance(mi)'].std()

    # Encode categorical variables
    us_accidents['Street_Type'] = us_accidents['Street'].apply(categorize_street)
    us_accidents['Highway_Flag'] = us_accidents['Street_Type'].apply(encode_street_type)
    us_accidents['Traffic_Signal_Flag'] = us_accidents['Traffic_Signal'].apply(encode_traffic_signal)
    us_accidents['Crossing_Flag'] = us_accidents['Crossing'].apply(encode_crossing)

    # Select desired columns
    selected_columns = ['Severity', 'Highway_Flag', 'Traffic_Signal_Flag', 'Crossing_Flag', 'Distance(mi)',
                        'Start_Hour_Sin', 'Start_Hour_Cos', 'Start_Month_Sin', 'Start_Month_Cos', 'Accident_Duration']

    us_accidents = us_accidents[selected_columns]

    print("first few rows of dataset\n", us_accidents.head())

    # Save preprocessed dataset
    preprocessed_path = 'data/us_accidents_data_cleaned.csv'
    us_accidents.to_csv(preprocessed_path, index=False)

    print(f"Preprocessed dataset saved at {preprocessed_path}")



# import numpy as np
# import pandas as pd
# import os


# state_abbrs = [
#     'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
#     'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
#     'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
# ]

# def categorize_street(street_name):

#     street_name = str(street_name)  # Ensure it's a string

#     # Check for highways
#     if any(x in street_name for x in ['I-', 'US-', 'State Route', 'SR-',
#                                       'Hwy', 'Highway','Fwy', 'Expressway', 'Expy',
#                                       'Turnpike', 'Pike', 'Pkwy',
#                                       'Interstate', 'Trwy', 'Tpke',
#                                       'Bypass', 'Corridor']) or \
#        any(f"{abbr}-" in street_name for abbr in state_abbrs):  # Detect state-specific highways
#         return 'Highway/Freeway'

#     # Check for local roads
#     elif any(x in street_name for x in ['Ave', 'Rd', 'St', 'Blvd', 'Dr', 'Ct', 'Pl', 'Ln', 'Way','Trail', 'Plaza']):
#         return 'Local Road'

#     # Default category
#     else:
#         return 'Other'

# def encode_street_type(street_type):
#     if pd.isna(street_type):
#         return 0  # Default to local road if missing

#     street_type = str(street_type)

#     # Highways/Freeways
#     if street_type == 'Highway/Freeway':
#         return 1  # Highway/Freeway
#     else:
#         return 0  # Local Road / Other

# def encode_traffic_signal(traffic_signal):
#     if pd.isna(traffic_signal):
#         return 0  # Default to no traffic signal if missing

#     traffic_signal = str(traffic_signal)

#     if traffic_signal == 'True':
#         return 1  # Traffic signal present
#     else:
#         return 0  # No traffic signal

# def encode_crossing(crossing):
#     if pd.isna(crossing):
#         return 0  # Default to no crossing if missing

#     crossing = str(crossing)

#     if crossing == 'True':
#         return 1  # Crossing present
#     else:
#         return 0  # No crossing

# # ============== NEW ADDITION ===============
# def preprocess_time_features(df):
#     # Convert time columns to datetime
#     df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
#     df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

#     # Extract features
#     df['Start_Hour'] = df['Start_Time'].dt.hour  # Hour of the day
#     df['Start_Month'] = df['Start_Time'].dt.month  # Month of the year
#     df['Accident_Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60  # Duration in minutes

#     # Convert time features into cyclic representation
#     df['Start_Hour_Sin'] = np.sin(2 * np.pi * df['Start_Hour'] / 24)
#     df['Start_Hour_Cos'] = np.cos(2 * np.pi * df['Start_Hour'] / 24)

#     df['Start_Month_Sin'] = np.sin(2 * np.pi * df['Start_Month'] / 12)
#     df['Start_Month_Cos'] = np.cos(2 * np.pi * df['Start_Month'] / 12)

#     return df


# print("Loading US Accidents dataset...")

# file_path = 'data/US_Accidents_March23.csv'

# # Use the skiprows parameter to load only the desired row
# step = 77
# # Generate a list of rows to skip
# rows_to_skip = [i for i in range(1, 7728394) if i % step != 0]  # Skip all except every 77th row

# # Read the CSV with the selected rows
# us_accidents = pd.read_csv(file_path, skiprows=rows_to_skip)

# # Verify the size of the sample
# print(f"Sample size: {len(us_accidents)}")

# print("Preprocessing the dataset...")

# # Select target + features with predictive power

# # ============== NEW ADDITION ===============
# # Add in time variables
# selected_columns = ['Severity', 'Traffic_Signal', 'Crossing', 'Street', 'Distance(mi)', 'Start_Time', 'End_Time']
# # ============== END OF NEW ADDITION ===============

# us_accidents = us_accidents[selected_columns]

# # Drop rows with missing values in selected columns
# us_accidents = us_accidents.dropna(subset=selected_columns)

# # ============== NEW ADDITION ===============
# # Preprocess time variables
# print("Start_Time: ", us_accidents['Start_Time'].iloc[0])
# print("End_Time: ", us_accidents['End_Time'].iloc[0])
# us_accidents = preprocess_time_features(us_accidents)
# print("First few rows of the dataset after time processing: ")
# print(us_accidents.head())
# # ============== END OF NEW ADDITION ===============

# # Encode categorical variables

# us_accidents['Street_Type'] = us_accidents['Street'].apply(categorize_street)
# us_accidents['Highway_Flag'] = us_accidents['Street_Type'].apply(encode_street_type)
# us_accidents['Traffic_Signal_Flag'] = us_accidents['Traffic_Signal'].apply(encode_traffic_signal)
# us_accidents['Crossing_Flag'] = us_accidents['Crossing'].apply(encode_crossing)
# us_accidents.drop(columns=['Street', 'Street_Type', 'Traffic_Signal', 'Crossing'], inplace=True)

# # Display the first few rows of the dataset
# print("First few rows of the dataset after categorical encoding: ")
# print(us_accidents.head())

# # Store preprocessed data into new csv file
# # Define folder and file paths
# folder_path = 'data'
# file_path = os.path.join(folder_path, 'us_accidents_data_cleaned.csv')

# # Create the folder if it doesn't exist
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

# # Save DataFrame to CSV
# us_accidents.to_csv(file_path, index=False)