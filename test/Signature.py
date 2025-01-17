import requests
import os
import pandas as pd
import gzip
import shutil
import scanpy as sc

# URLs for the files
phenotypes_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE119nnn/GSE119911/suppl/GSE119911_mixed_sample_phenotypes.xlsx"
tmp_matrix_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE119nnn/GSE119911/suppl/GSE119911_All_Merge_umi_tpm_gene.txt.gz"

# Local filenames
phenotypes_file = "/Users/aryamaanbose/Desktop/Statescope-dev/Statescope/test/GSE119911_mixed_sample_phenotypes.xlsx"
tmp_matrix_file_gz = "/Users/aryamaanbose/Desktop/Statescope-dev/Statescope/test/GSE119911_All_Merge_umi_tpm_gene.txt.gz"
tmp_matrix_file = "/Users/aryamaanbose/Desktop/Statescope-dev/Statescope/test/GSE119911_All_Merge_umi_tpm_gene.txt"

# Step 1: Download the phenotype file
print("Downloading phenotypes file...")
response = requests.get(phenotypes_url, stream=True)
with open(phenotypes_file, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)
print(f"Phenotypes file downloaded: {phenotypes_file}")

# Step 2: Download the expression matrix file
print("Downloading expression matrix file...")
response = requests.get(tmp_matrix_url, stream=True)
with open(tmp_matrix_file_gz, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)
print(f"Expression matrix file downloaded: {tmp_matrix_file_gz}")

# Step 3: Extract the gzipped expression matrix
print("Extracting expression matrix file...")
with gzip.open(tmp_matrix_file_gz, "rb") as f_in:
    with open(tmp_matrix_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
print(f"Expression matrix file extracted: {tmp_matrix_file}")

# Step 4: Load phenotype and expression matrix
print("Loading phenotype file...")
phenotypes = pd.read_excel(phenotypes_file)
print("Phenotypes file loaded.")

print("Loading expression matrix file...")
expression_data = pd.read_csv(tmp_matrix_file, sep="\t", index_col=0)
print("Expression matrix file loaded.")

####Cell type columns 
# Step 5: Prepare phenotype data
if "Type" in phenotypes.columns:
    phenotypes.rename(columns={"Type": "celltype"}, inplace=True)
else:
    raise ValueError("The phenotype file does not contain a 'Type' column.")

print("Available columns in phenotypes:")
print(phenotypes.columns)

print("First few rows of the phenotype DataFrame:")
print(phenotypes.head())

phenotypes.set_index("Sample.2", inplace=True)  # Replace 'cell_id' with the actual column name for cell IDs

# Step 6: Ensure alignment between expression and phenotype data
common_cells = expression_data.columns.intersection(phenotypes.index)
expression_data = expression_data[common_cells]
phenotypes = phenotypes.loc[common_cells]

# Step 7: Create AnnData object
Signature = sc.AnnData(X=expression_data.T.values, obs=phenotypes)

# Step 8: Save AnnData object
Signature.write("/Users/aryamaanbose/Desktop/Statescope-dev/Statescope/test/GSE119911_single_cell.h5ad")
print("AnnData object created and saved as 'GSE119911_single_cell.h5ad'.")
