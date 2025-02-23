import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("US_Accidents_Cleaned.csv")

# Convert 'End_Time' to datetime format (handling errors)
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

# Extract 'End_Hour', ensuring missing values are dropped
df["End_Hour"] = df["End_Time"].dt.hour

# Drop NaN values and convert to integer
df = df.dropna(subset=["End_Hour"])
df["End_Hour"] = df["End_Hour"].astype(int)

# Check if we have valid data
if df["End_Hour"].empty:
    print("No valid End_Hour data available. Check the dataset!")
else:
    # Plot histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(df["End_Hour"], bins=24, kde=False, discrete=True)

    # Add labels
    plt.xlabel("Hour of the Day")
    plt.ylabel("Frequency")
    plt.title("Distribution of Accidents by End Hour")

    plt.show()
