import csv
import re
import ast

# File paths
input_txt = 'input.txt'
output_csv = 'flattened_data.csv'

# Prepare storage
entries = []

with open(input_txt, 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_image_id = None
responses = []
path = None

for line in lines:
    line = line.strip()
    
    if line.startswith("Response for"):
        # Save previous entry if exists
        if current_image_id and responses and path:
            for r in responses:
                entries.append([current_image_id, r['question'], r['answer'], path])
        
        # Parse new image ID and response list
        match = re.match(r"Response for ([^ ]+) ?: (.+)", line)
        if match:
            current_image_id = match.group(1)
            responses = ast.literal_eval(match.group(2))  # safe eval of list
    elif line.startswith("Last record:"):
        record = ast.literal_eval(line.replace("Last record: ", ""))
        path = record.get('path')

# Add last entry
if current_image_id and responses and path:
    for r in responses:
        entries.append([current_image_id, r['question'], r['answer'], path])

# Write to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'question', 'answer', 'path'])
    writer.writerows(entries)

print(f"Flattened data written to {output_csv}")
