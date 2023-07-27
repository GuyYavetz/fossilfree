import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
import numpy as np


# Set up Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)
client = gspread.authorize(creds)
sheet = client.open('#USE YOUR SHEET NAME HERE')
worksheet = sheet.get_worksheet(0)


B_values = np.array([float(val.replace(',', ''))
                    if val else 0 for val in worksheet.col_values(9)[1:]])
A_values = np.array([float(val.replace(',', ''))
                    if val else 0 for val in worksheet.col_values(10)[1:]])


def normalize_results(A, B):
    A = np.where(A == 0, 1, A)
    B = np.where(B == 0, 1, B)
    # B = np.where(B > A, A, B)
    A = A + 1e-10
    C = B / A
    C = np.log(C + 1e-10)
    return C


# Calculate all normalized values
C_values = normalize_results(A_values, B_values)

# Define batch size
batch_size = 10

for i in range(0, len(C_values), batch_size):
    C_batch = C_values[i:i+batch_size]

    # Construct a list of cell objects which we will update in batch
    # +2 to account for 1-indexing and header row
    cell_list = worksheet.range('M{}:M{}'.format(
        i+2, min(i+batch_size+2, len(C_values)+2)))

    # Assign new value to each cell in the batch
    for cell, value in zip(cell_list, C_batch):
        cell.value = value

    # Update in batch
    worksheet.update_cells(cell_list)
