import zipfile

zip_path = "new-plant-diseases-dataset.zip"
extract_path = "plant_disease_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction completed!")