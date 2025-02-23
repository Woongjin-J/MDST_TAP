import pandas as pd
import matplotlib.pyplot as plt

# TODO: Load dataset; replace this csv to your file
df = pd.read_csv("US_Accidents_Cleaned.csv")

# Step 1: Handling the format
# TODO: Remove extra precision if exists


# TODO: Extract relevant time-based features
# df["Year"] = df["Start_Time"].dt.year
# df["Month"] = df["Start_Time"].dt.month
# df["Day"] = df["Start_Time"].dt.day
# df["Hour"] = df["Start_Time"].dt.hour
# df["Weekday"] = df["Start_Time"].dt.weekday  # 0 = Monday, 6 = Sunday
# df["Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)  # 1 = Weekend, 0 = Weekday


# TODO: Fix missing values for numerical columns
# TODO: Ensure Severity is numeric
# Fill numerical missing values with the median
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical missing values with mode (most frequent value)
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert Severity to numeric (if it's not)
df["Severity"] = pd.to_numeric(df["Severity"], errors="coerce")

# Check for any NaN values in Severity
print(df["Severity"].isnull().sum())  # If NaNs exist, fill them with mode
df["Severity"] = df["Severity"].fillna(df["Severity"].mode()[0])



# Pie Charts
severity_counts = df['Severity'].value_counts(normalize=True) * 100

plt.figure(figsize=(7, 7))
plt.pie(
    severity_counts, labels=severity_counts.index, autopct="%1.1f%%", startangle=140
)
plt.title("Accident Severity Distribution")
plt.show()


# Road conditions presence
road_conditions = ['Crossing', 'Traffic_Signal', 'Junction']
for condition in road_conditions:
  plt.figure(figsize=(6, 4))
  df[condition].value_counts().plot(kind='bar', color=['blue', 'orange'])
  plt.title(f'Accidents at {condition} Presence')
  plt.xlabel(f'{condition} Present (0=No, 1=Yes)')
  plt.ylabel('Number of Accidents')
  plt.show()

# Bar Plots
# Accident Cases vs Hours
hourly_counts = df['Hour'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(hourly_counts.index, hourly_counts.values, color='blue')
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.title("Accident Cases by Hour")
plt.xticks(range(0, 24))
plt.show()


# Accident Cases vs Months
monthly_counts = df['Month'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(monthly_counts.index, monthly_counts.values, color='green')
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.title("Accident Cases by Month")
plt.xticks(range(1, 13))
plt.show()

# Accident Cases vs Different Temperature
df["Temperature_Bins"] = pd.cut(df["Temperature(F)"], bins=10)
temp_counts = df["Temperature_Bins"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(temp_counts.index.astype(str), temp_counts.values, color='red')
plt.xlabel("Temperature (F) Ranges")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Temperature")
plt.xticks(rotation=45)
plt.show()

# Accident Cases vs Different Humidity
df["Humidity_Bins"] = pd.cut(df["Humidity(%)"], bins=10)
humidity_counts = df["Humidity_Bins"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(humidity_counts.index.astype(str), humidity_counts.values, color='purple')
plt.xlabel("Humidity (%) Ranges")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Humidity")
plt.xticks(rotation=45)
plt.show()

# Accident Cases vs Wind Speed
df["WindSpeed_Bins"] = pd.cut(df["Wind_Speed(mph)"], bins=10)
wind_counts = df["WindSpeed_Bins"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(wind_counts.index.astype(str), wind_counts.values, color='orange')
plt.xlabel("Wind Speed (mph) Ranges")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Wind Speed")
plt.xticks(rotation=45)
plt.show()


df["Start_Hour"] = df["Start_Time"].dt.hour