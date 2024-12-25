import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gdown
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import os





# File ID from the Google Drive link
file_id = '1xFesb_-mjQU2ZX1Sk7slCkTXBKCUaohZ' # ID of the File usually after this ---> https://drive.google.com/file/d/ <--and before this part of the URL --> view?usp=sharing <--
# New smallaer file id
# file_id = '1ECMVWnoCk6FzZBYPEXo9UR_BNqzSLCBj'

# Construct the download URL
download_url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
gdown.download(download_url, 'data.csv', quiet=False)

path="data.csv"
# path = "FeedBack_Dataset_2.csv"

df = pd.read_csv(path)

print("Dataset preview:")
print(df.head())

# Load the dataset
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Display the original data structure
print("Original Data:")
print(data.head())

# Step 1: Standardize column names
data.columns = data.columns.str.strip()  # Remove extra spaces

# Step 2: Convert Yes/No responses to binary values (1 for Yes, 0 for No)
binary_mapping = {"Yes": 1, "No": 0}
data.replace(binary_mapping, inplace=True)

# Step 3: Display the cleaned dataset
print("\nCleaned Data:")
print(data.head())

# Save the cleaned dataset for future use
cleaned_file_path = 'Cleaned_FeedBack_Dataset.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved as {cleaned_file_path}.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = 'Cleaned_FeedBack_Dataset.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Step 1: Define question groups
antenatal_cols = [col for col in data.columns if col.startswith("Antenatal_")]
postpartum_cols = [col for col in data.columns if col.startswith("Postpartum_")]

# Step 2: Calculate percentage scores
data['Antenatal_Score_Percent'] = data[antenatal_cols].sum(axis=1) / len(antenatal_cols) * 100
data['Postpartum_Score_Percent'] = data[postpartum_cols].sum(axis=1) / len(postpartum_cols) * 100

# Step 3: Plot percentage scores
plt.figure(figsize=(14, 6))

# Antenatal Score Percentage
plt.subplot(1, 2, 1)
sns.histplot(data['Antenatal_Score_Percent'], kde=True, bins=10, color='blue')
plt.title('Antenatal Depression Percentage Scores')
plt.xlabel('Antenatal Depression Score (%)')
plt.ylabel('Frequency')

# Postpartum Score Percentage
plt.subplot(1, 2, 2)
sns.histplot(data['Postpartum_Score_Percent'], kde=True, bins=10, color='green')
plt.title('Postpartum Depression Percentage Scores')
plt.xlabel('Postpartum Depression Score (%)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 4: Group by age and calculate average percentage scores
bins = [0,20, 25, 30, 35, 40, 100]  # Age ranges
labels = ['<20','20-25', '25-30', '30-35', '35-40', '40+']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

age_group_scores = data.groupby('Age_Group')[['Antenatal_Score_Percent', 'Postpartum_Score_Percent']].mean()

# Step 5: Visualize age-based percentage scores
age_group_scores.plot(kind='bar', figsize=(10, 6), color=['blue', 'green'])
plt.title('Average Depression Percentage Scores by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Score (%)')
plt.legend(['Antenatal (%)', 'Postpartum (%)'])
plt.xticks(rotation=0)
plt.show()

# Print results
print("\nAverage Depression Percentage Scores by Age Group:")
print(age_group_scores)








# Define directory for saving models
model_save_path = './saved_models/'

# Create the directory if it doesn't exist
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    print(f"Directory '{model_save_path}' created for saving models.")

# Load and prepare the dataset
file_path = 'Cleaned_FeedBack_Dataset.csv'
data = pd.read_csv(file_path)

# Define columns for Antenatal and Postpartum depression
antenatal_cols = [col for col in data.columns if col.startswith("Antenatal_")]
postpartum_cols = [col for col in data.columns if col.startswith("Postpartum_")]

# Calculate percentage scores for Antenatal and Postpartum Depression
data['Antenatal_Score_Percent'] = data[antenatal_cols].sum(axis=1) / len(antenatal_cols) * 100
data['Postpartum_Score_Percent'] = data[postpartum_cols].sum(axis=1) / len(postpartum_cols) * 100

# Define Depression Levels
def classify_depression(score):
    if score < 33:
        return 0  # Low
    elif score < 66:
        return 1  # Moderate
    else:
        return 2  # Severe

data['Antenatal_Level'] = data['Antenatal_Score_Percent'].apply(classify_depression)
data['Postpartum_Level'] = data['Postpartum_Score_Percent'].apply(classify_depression)

# Group ages into bins
bins = [0, 20, 25, 30, 35, 40, np.inf]
labels = ['<20', '20-24', '25-29', '30-34', '35-39', '40+']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)

# One-hot encode Age_Group
encoder = OneHotEncoder(sparse_output=False)
age_group_encoded = encoder.fit_transform(data[['Age_Group']])
age_group_columns = encoder.get_feature_names_out(['Age_Group'])
age_group_df = pd.DataFrame(age_group_encoded, columns=age_group_columns, index=data.index)

# Add to the dataset
data = pd.concat([data, age_group_df], axis=1)

# Features and target
X = pd.concat([data[antenatal_cols], age_group_df], axis=1)
y = data['Postpartum_Level']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=10, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
}

# Create directory for saving models if not already done
import os
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    print(f"Directory '{model_save_path}' created for saving models.")

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)

    # Save the model
    model_file = model_save_path + f"{name.replace(' ', '_')}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name} model to {model_file}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.title(f"{name} - Confusion Matrix\nAccuracy: {acc:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Print accuracy and evaluation
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "Moderate", "Severe"]))

