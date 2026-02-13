import csv
import os
import requests
import pandas as pd
import pdb
import certifi
from tqdm import tqdm
import urllib3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

csv_file_path = "building_dataset.csv"  # Path to your CSV file with the URLs
download_folder = "downloaded_images"  # Folder where the downloaded images will be saved
threads = 10  # Number of threads for concurrent downloading

# Disable the InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_jpg(row):

    # session = requests.Session()
    # session.verify = certifi.where()
    # a = HTTPAdapter(max_retries=3)
    # session.mount("https://", a)

    filename = f"{row['PID']}.jpg"
    url = row['Image']
    file_path = os.path.join(download_folder, filename)
    
    if os.path.exists(filename):
        print(f"File already exists, skipping download: {filename}")
    else:
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)
                # for chunk in response.iter_content(chunk_size=8192):
                #     f.write(chunk)

            # print(f"Downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}: {e}")

            return url, None
    return url, file_path
    # session.close() 

# def download_jpg_from_csv(csv_file_path, download_folder):
#     session = requests.Session()
#     session.verify = certifi.where()
#     a = HTTPAdapter(max_retries=3)
#     session.mount("https://", a)

#     for index, row in tqdm(df.iterrows()):
#         # pdb.set_trace()
#         filename = f"{row['PID']}.jpg"
#         url = row['Image']
#         file_path = os.path.join(download_folder, filename)
        
#         if os.path.exists(file_name):
#             print(f"File already exists, skipping download: {file_name}")
#         else:
#             try:
#                 response = session.get(url, verify=False)
#                 response.raise_for_status()

#                 with open(file_path, 'wb') as f:
#                     f.write(response.content)
#                     # for chunk in response.iter_content(chunk_size=8192):
#                     #     f.write(chunk)

#                 print(f"Downloaded: {filename}")
#             except requests.exceptions.RequestException as e:
#                 print(f"Failed to download {filename}: {e}")
#     session.close() 


if __name__ == "__main__":
    # Modify the paths accordingly

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    df = pd.read_csv(csv_file_path, usecols=['PID','Image'])

    with ThreadPoolExecutor(max_workers=threads) as exe:
        # dispatch all download tasks to worker threads
        futures = [exe.submit(download_jpg, row) for _,row in df.iterrows()]
        # report results as they become available
        for future in as_completed(futures):
            # retrieve result
            link, outpath = future.result()
            # check for a link that was skipped
            if outpath is None:
                print(f'>skipped {link}')
            else:
                print(f'Downloaded {link} to {outpath}')

    # executor = ThreadPoolExecutor(max_workers=threads)
    # with executor:
    #     results = executor.map(download_jpg, df.iterrows())

    # download_jpg_from_csv(csv_file_path, download_folder)