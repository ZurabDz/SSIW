import subprocess
import zipfile

token = "16pmWKSZYvmuocHCXbcMKA8HVlfKvFoet"
download_command = f"gdown {token}"

subprocess.run(download_command.split())

with zipfile.ZipFile("data.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
