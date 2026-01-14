import pandas as pd
import requests

# --- CONFIGURATION ---
INPUT_FILE = "california_questions.xlsx"
OUTPUT_FILE = "answers_output.xlsx"
API_URL = "http://0.0.0.0:8000/api/agents/generic_chatbot/"

# 1. Load the Excel file
df = pd.read_excel(INPUT_FILE)

# Assuming your Excel has a column named 'Questions'
# If it's different, change 'Questions' to your actual column name
results = []

print(f"Starting API processing for {len(df)} rows...")

# 2. Iterate through each row and call the API
for index, row in df.iterrows():
    query_text = str(row['question'])
    
    payload = {'query': query_text}
    headers = {}
    
    try:
        # Make the POST request
        response = requests.post(API_URL, data=payload, headers=headers)
        
        if response.status_code == 200:
            # Append the response text (or JSON field)
            results.append(response.text)
            print(f"Processed row {index + 1} successfully.")
        else:
            results.append(f"Error: {response.status_code}")
            print(f"Failed at row {index + 1}: {response.status_code}")
            
    except Exception as e:
        results.append(f"Exception: {str(e)}")
        print(f"Exception at row {index + 1}: {e}")

# 3. Add the results to a new column and save
df['Chatbot_Response'] = results
df.to_excel(OUTPUT_FILE, index=False)

print(f"\nDone! All results saved to {OUTPUT_FILE}")