# Fossil-Free

### Description
#### This script fetches a list of company names from a Google Sheet, uses OpenAI's GPT-3 model to generate information about each company, and then updates the Google Sheet with this generated information.

1. Import necessary modules: This script imports a number of Python libraries, including `gspread` for interacting with Google Sheets, `oauth2client` for authorizing the script to access Google services, `os` for operating system related tasks, `openai` for generating text using OpenAI's API, and `time` for time-related functions.

2. Set up Google Sheets API credentials: The script then sets up the authorization credentials to use Google Sheets API. The details are stored in a JSON file named 'client_s.json', and it's reading the credentials from that file.

3. Open Google Sheets: The script opens a specific Google Sheet with the name 'fossil_fuel_data'. 

4. Fetch data from Google Sheets: The script fetches data from column 5 (company names) and column 15 (company data) from the first sheet in the Google Sheets document.

5. Equalize lengths of lists: The script ensures that the `company_names` list and the `company_data` list have the same length. If `company_data` is shorter, the script appends empty strings to it.

6. Set OpenAI API key: The script sets the API key for OpenAI. 

7. Loop over company names: For each company in the list, the script:
    - Checks if there is already data about the company in the corresponding cell of the `company_data` list. 
    - If there isn't, it creates a prompt asking for information about the company and feeds this prompt to OpenAI's API.
    - The response from OpenAI's API is then stored in `company_data` and also added to the `cells_to_update` list, which keeps track of the cells that need to be updated in Google Sheets.

8. Handle OpenAI's rate limits: If the OpenAI API rate limit is exceeded, the script will pause for 60 seconds before moving on to the next company.

9. Batch updates to Google Sheets: Every time the `cells_to_update` list has 10 items, the script pushes these updates to the Google Sheet and clears the list. 

10. Update any remaining cells: After the loop over all companies, the script updates any remaining cells in Google Sheets that are in the `cells_to_update` list.

### Google
#### This script fetches a list of company names from a Google Sheet, performs a Google search for each company, extracts the number of search results, and updates the Google Sheet with this data. It performs these operations in batches for efficiency and to minimize the risk of running into rate limits or other issues.

1. Import necessary modules: The script imports the Python libraries necessary for its functionality. These include `selenium` for automating a web browser, `time` and `re` for handling time and regular expressions respectively, `gspread` and `oauth2client` for accessing Google Sheets, and `os` and `openai` for various other tasks.

2. Set up Google Sheets API credentials: Similar to the previous code, this script sets up the authorization credentials for accessing Google Sheets API.

3. Open Google Sheets: It opens a specific Google Sheet and retrieves the company names from the fifth column of the first worksheet.

4. Set up WebDriver: The script sets up a WebDriver using Chrome's WebDriver (you need to specify your driver path). This is what allows it to automate a web browser. A WebDriverWait object is also set up to allow the script to pause until certain conditions are met.

5. Search Google for each company: For each company in the list, the script automates a Google search for that company. It does this in batches of 10 companies at a time, starting from the 542nd company.

    - Strip trailing periods from the company name.
    - Load the Google homepage.
    - Find the search box and input the company name.
    - Perform a search (simulates pressing the Return key).
    - Wait for the search results page to load.

6. Extract search results count: Once the results page has loaded, the script tries to find and extract the number of search results from the page.

    - It does this by locating an element with the ID 'result-stats' and extracting its text. This element contains the number of search results.
    - It uses a regular expression to extract the actual number from the text.
    - If successful, it prints the company name and the number of results, and adds the number to the batch results.
    - If the element can't be found, it prints an error message and skips to the next company.

7. Update Google Sheets: After it has processed a batch of companies, the script updates the corresponding cells in the Google Sheet with the number of search results for each company.

8. Quit WebDriver: Once all the companies have been processed, the script quits the WebDriver.

### Google_RenewableEnergy
#### This script fetches a list of company names from a Google Sheet, performs a Google search for each company along with the phrase 'Renewable energy', extracts the number of search results, and updates the Google Sheet with this data. The operations are performed in batches for efficiency and to minimize the risk of running into rate limits or other issues.

1. Import necessary modules: As with the previous scripts, the necessary Python libraries are imported. These include `selenium` for web browser automation, `time` and `re` for handling time and regular expressions respectively, `gspread` and `oauth2client` for accessing Google Sheets, and `os` and `openai` for other tasks.

2. Set up Google Sheets API credentials: The script sets up the authorization credentials for accessing Google Sheets API.

3. Open Google Sheets: The script opens a specific Google Sheet and retrieves company names from the fifth column of the first worksheet.

4. Set up WebDriver: It sets up a WebDriver using Chrome's WebDriver (you need to specify your driver path). This is what allows it to automate a web browser. It also sets up a WebDriverWait object that allows the script to pause until certain conditions are met.

5. Search Google for each company with a specific keyword: For each company in the list, the script automates a Google search for that company, in addition to the phrase 'Renewable energy'. It does this in batches of 10 companies at a time, starting from the first company.

    - It strips trailing and leading periods from the company name.
    - It loads the Google homepage.
    - It finds the search box, inputs the company name followed by 'Renewable energy', and performs a search.
    - It waits for the search results page to load.

6. Extract search results count: Once the results page has loaded, the script extracts the number of search results from the page.

    - It finds the element with the ID 'result-stats' and extracts its text. This element contains the number of search results.
    - It uses a regular expression to extract the actual number from the text.
    - If successful, it prints the company name and the number of results, and adds the number to the batch results.
    - If a `StaleElementReferenceException` is encountered while trying to find the search box, it skips to the next company.

7. Update Google Sheets: After it has processed a batch of companies, it updates the corresponding cells in the Google Sheet with the number of search results for each company. The results are put in column 'Q'.

8. Quit WebDriver: Once all the companies have been processed, the script quits the WebDriver.

### Google Normalizion
#### This script retrieves data from a Google Sheet(The coulmns we create in Google and Google_RenewableEnergy), performs a normalization operation on the data, and writes the results back to the Google Sheet

1. Import necessary modules: This script imports the required Python libraries. These include `gspread` for accessing Google Sheets, `oauth2client` for authentication, and `numpy` for numerical operations.

2. Set up Google Sheets API credentials: The script sets up the authorization credentials for accessing the Google Sheets API.

3. Open Google Sheets: The script opens a specific Google Sheet and retrieves the first worksheet.

4. Retrieve data from Google Sheets: The script retrieves the data from columns B (index 9) and A (index 10) of the worksheet and converts them to floating point numbers. Commas in the numbers are removed. Any cell that does not contain a number is assumed to contain zero.

5. Define a normalization function: The function `normalize_results` takes two `numpy` arrays, A and B, and returns a new array C where each element is the natural logarithm of the corresponding element in B divided by the corresponding element in A. This function also ensures that division by zero is avoided.

6. Normalize the data: The script uses the `normalize_results` function to normalize the values retrieved from the Google Sheets.

7. Update Google Sheets: The script then writes these normalized values back to the Google Sheet. The update is done in batches of 10 cells at a time for efficiency. The results are written to column M of the Google Sheet.


### Few_Shot_Train
#### This script uses OpenAI's language model for a classification task with data sourced from a Google Sheet(The Description provided earlier with OpenAI's language model)

1. Import necessary modules: The script imports the required Python libraries, which includes `gspread` for Google Sheets interaction, `oauth2client` for Google Sheets authorization, `openai` for using OpenAI's language models, `time` for delaying execution, and `matplotlib.pyplot` and `seaborn` for potential data visualization (not used in this script).

2. Set up Google Sheets API credentials: Similar to the earlier script, it sets up the authorization credentials for accessing the Google Sheets API.

3. Open Google Sheets: It opens a specific Google Sheet and retrieves the first worksheet.

4. Set OpenAI API key: The script sets up the OpenAI API key. This allows the script to make requests to OpenAI's language model.

5. Retrieve company descriptions: It retrieves the company descriptions from the Google Sheet's column 'O' (which is the 15th column).

6. Define example associations: A list of examples is defined, where each example is a tuple consisting of a company description and its association with renewable energy.

7. Classify companies: The script loops over all the company descriptions retrieved from the Google Sheet. For each description, it constructs a prompt based on the examples and uses OpenAI's language model to classify the company's association with renewable energy. The classification result is then written back to the Google Sheet.

8. Delay execution: A short delay is introduced between each iteration to avoid hitting the API rate limit of OpenAI's language model.


### Few_Shot_Evaluation
####  This script allows you to assess the performance of a classification task by comparing the model's predictions against true labels, and visualizing the results in a confusion matrix.

1. Import necessary modules: The script imports the required Python libraries. The `gspread` library is used for interacting with Google Sheets, `oauth2client` for Google Sheets authorization, `matplotlib.pyplot` and `seaborn` for data visualization, `sklearn.metrics` for computing the classification report and confusion matrix, and `numpy` for handling arrays.

2. Set up Google Sheets API credentials: The script sets up the authorization credentials for accessing the Google Sheets API. 

3. Open Google Sheet: The specific Google Sheet is opened, and the first worksheet is retrieved.

4. Retrieve true and predicted labels: It retrieves the true labels (manually assigned labels) and the predicted labels (labels predicted by the model) from the Google Sheet. The labels are taken from the columns 'H' and 'K' (8th and 11th columns) respectively.

5. Display unique labels: The script prints the unique labels present in both the true and predicted labels. This helps to understand the types of labels that the classification task has handled.

6. Compute and print the classification report: It computes the classification report using `classification_report` from `sklearn.metrics`, which includes precision, recall, f1-score, and support for each class.

7. Compute confusion matrix: The confusion matrix is computed using `confusion_matrix` from `sklearn.metrics`. The confusion matrix is a table that is often used to describe the performance of a classification model.

8. Plot and display the confusion matrix: The confusion matrix is plotted as a heatmap using `seaborn`'s `heatmap` function, and then displayed. This visualization provides an intuitive representation of the classifier's performance.

### Tf-idf
####  This script handles the task of text classification on company descriptions, using SVM as the classifier, and evaluates the performance of the classification task by comparing the predicted labels with the true labels

1. **Import necessary modules:** This script requires `gspread` for Google Sheets interaction, `oauth2client` for Google Sheets authorization, `sklearn` for machine learning tasks and performance evaluation, and `matplotlib` for data visualization.

2. **Google Sheets API credentials setup:** The script initializes the authorization credentials to access Google Sheets API.

3. **Open Google Sheet:** It opens the required Google Sheet and retrieves the first worksheet.

4. **Retrieve company descriptions:** Company descriptions are retrieved from the Google Sheet from column 'G' (7th column).

5. **Split data into training and testing sets:** The descriptions are divided into training and testing sets.

6. **Retrieve labels:** The labels for the classification task are retrieved from the Google Sheet from column 'H' (8th column).

7. **Feature extraction:** It performs feature extraction on the company descriptions using TF-IDF Vectorizer, which converts text data into numerical vectors that can be used for machine learning tasks.

8. **Train SVM classifier:** A Support Vector Machine (SVM) classifier with a linear kernel is trained using the TF-IDF vectors and the corresponding labels.

9. **Predict labels for test data:** The trained classifier is used to predict labels for the test data.

10. **Write predictions to Google Sheet:** The predicted labels are written back to the Google Sheet in batches of 10, in column 'L' (12th column).

11. **Calculate and display accuracy of classifier:** The script retrieves the true labels for the test set from the Google Sheet, then calculates and displays the accuracy of the classifier by generating a confusion matrix using `plot_confusion_matrix` from `sklearn.metrics`, and a classification report using `classification_report` from `sklearn.metrics`. The confusion matrix is displayed as a plot using matplotlib, and the classification report is printed.


### LogisticRegression_Tf-Idf
####  This script performs a classification task using Logistic Regression. It takes into account multiple features ('description', 'Section Working In - English', 'Normalization') and processes each of them using a different transformation (TF-IDF Vectorizer, OneHotEncoder, and MinMaxScaler respectively).

1. **Import necessary modules:** This script requires `pandas`, `gspread`, `oauth2client`, `sklearn`, `seaborn`, and `matplotlib`. 

2. **Google Sheets API credentials setup:** It initializes the authorization credentials to access Google Sheets API and opens the required Google Sheet.

3. **Retrieve and preprocess the data:** It retrieves the data from Google Sheet and converts it into a pandas DataFrame. Next, it casts the 'Normalization' column to numeric and specifies the input features (`X`) and the target variable (`y`).

4. **Split the data into training and testing sets:** It uses `train_test_split` function from `sklearn.model_selection` to split the data into training and testing sets.

5. **Set up the preprocessing and classification pipeline:** The `ColumnTransformer` from `sklearn.compose` is used to specify different preprocessing steps for different types of features. For text data, it uses `TfidfVectorizer`, for categorical data it uses `OneHotEncoder`, and for numerical data it uses `MinMaxScaler`. These preprocessing steps are followed by a `LogisticRegression` classifier. The entire sequence is encapsulated into a `Pipeline`.

6. **Train the classifier:** It fits the pipeline to the training data.

7. **Evaluate the classifier:** It computes the classification accuracy on the test set and prints it out.

8. **Predict the labels of the test set:** It uses the trained classifier to predict labels for the test set.

9. **Compute and display the confusion matrix:** It computes the confusion matrix for the true and predicted labels of the test set, and displays it using seaborn and matplotlib.

10. **Generate and print the classification report:** It generates a classification report which includes precision, recall, f1-score, and support for each class, using the `classification_report` function from `sklearn.metrics`, and prints it out.


