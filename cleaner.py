import os
import re

INPUT_DIR = "scraped_articles"
OUTPUT_DIR = "cleaned_articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Patterns to remove from scraped content
date_pattern = re.compile(r"\b\d{1,2} \w{3}, \d{1,2}:\d{2} [APM]{2}\b", re.IGNORECASE)
views_pattern = re.compile(r"^\d+\s+Views?$", re.IGNORECASE)
comments_pattern = re.compile(r"^\d+\s+Comments?$", re.IGNORECASE)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.strip()

        # Stop at feedback section
        if "Was this article helpful?" in line:
            break

        if not line:
            continue

        # Skip metadata lines
        if date_pattern.match(line) or views_pattern.match(line) or comments_pattern.match(line):
            continue

        cleaned_lines.append(line)

    # Create title from filename
    title = filename.replace(".txt", "").replace("-", " ").replace("_", " ").strip()
    cleaned_text = f"# {title}\n\n" + "\n".join(cleaned_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Cleaned: {filename}")