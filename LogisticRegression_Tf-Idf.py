import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)

client = gspread.authorize(creds)
sheet = client.open('#USE YOUR SHEET NAME HERE')
worksheet = sheet.get_worksheet(0)

# Read the data from the Google Sheet
data = worksheet.get_all_values()
header = data[0]  
data = data[1:]  

df = pd.DataFrame(data, columns=header)

# Column names
description = 'description'
section = 'Section Working In - English'
normalization = 'Normalization'
label = 'label'

df[normalization] = pd.to_numeric(df[normalization])

X = df[[description, section, normalization]]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(), description),
        ('ohe', OneHotEncoder(), [section]),
        ('scaler', MinMaxScaler(), [normalization])
    ])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


report = classification_report(y_test, y_pred)
print(report)
