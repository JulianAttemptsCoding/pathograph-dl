import csv
import pycountry
from pathlib import Path

# Create the directory if it doesn't exist
Path("data/processed/meta").mkdir(parents=True, exist_ok=True)

# Read the ISO3 codes from your file
with open("config/countries_194_iso3.txt", "r") as f:
    iso3_codes = [line.strip().upper() for line in f if line.strip()]

# Create node_index.csv with proper columns
with open("data/processed/meta/node_index.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "iso3", "name", "region"])  # Header

    for i, iso3 in enumerate(iso3_codes):
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            name = country.name if country else iso3
            # You can add region information if available
            writer.writerow([i, iso3, name, ""])
        except Exception as e:
            print(f"Error processing {iso3}: {e}")
            writer.writerow([i, iso3, iso3, ""])

print(f"node_index.csv created with {len(iso3_codes)} nodes")