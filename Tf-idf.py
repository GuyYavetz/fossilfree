import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Set up Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)
client = gspread.authorize(creds)
sheet = client.open('#USE YOUR SHEET NAME HERE')
worksheet = sheet.get_worksheet(0)

# Retrieve the company descriptions from the sheet
descriptions = worksheet.col_values(7)  

# Separate descriptions into training and testing sets
train_descriptions = descriptions[:211]
test_descriptions = descriptions[211:]

# Retrieve the labels from the sheet
labels = worksheet.col_values(8)[:211]

# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_descriptions)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, labels)

# Now use the trained classifier to predict labels for the unlabeled companies
test_vectors = vectorizer.transform(test_descriptions)
predictions = classifier_linear.predict(test_vectors)

# Now let's write the predictions back to the Google Sheet in batches of 10
for i in range(0, len(predictions), 10):
    # Assuming row numbering starts from 1, not 0
    cell_list = worksheet.range(
        'L' + str(i+212) + ':L' + str(min(i+222, len(predictions)+212)))
    for cell, prediction in zip(cell_list, predictions[i:i+10]):
        cell.value = prediction
    worksheet.update_cells(cell_list)

# Now let's calculate the accuracy of the classifier
test_labels = worksheet.col_values(8)[211:]
plt.figure(figsize=(10, 7))
plot_confusion_matrix(classifier_linear, test_vectors, test_labels)
plt.show()

report = classification_report(test_labels, predictions)
print(report)
