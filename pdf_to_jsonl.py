import pymupdf4llm
import json
import os
from tqdm import tqdm

def folder_to_jsonl(pdf_folder, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        # Loop through every file in your folder
        for filename in tqdm(os.listdir(pdf_folder)):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_folder, filename)
                try:
                    # Convert PDF to Markdown text
                    md_text = pymupdf4llm.to_markdown(filepath)
                    
                    # Each PDF becomes one line in the JSONL file
                    json_line = json.dumps({"text": md_text}, ensure_ascii=False)
                    f.write(json_line + "\n")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Run the extraction
folder_to_jsonl("./driving_license_and_license_renewal_pdfs 1/driving_license_and_license_renewal_pdfs/date_dec_4", "corpus.jsonl")