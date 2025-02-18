import os
import json

# Directory containing the JSON files
directory = './data/Movie_top100'

# Traverse through the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)

        # Read and modify the JSON file
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Add 'filename' key to each dictionary
        modified_lines = []
        for line in lines:
            d = json.loads(line)
            d["filename"] = filename
            modified_lines.append(json.dumps(d))

        # Write the modified content back to the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(modified_lines))

print("Preprocessing completed for all JSON files.")