# City-Transformers
Repository for Nature Cities paper.

- `data/`
  - `City_Boundary.geojson`
    - Description: polygon representing city boundaries for Worcester, MA
  - `Buildings_with_addr.geojson`
    - Description: list of 44,059 (OBJECTIDs, TYPE, ..., address, geometry)
    - Note: Filtered from `Buildings.geojson` to inlcude only TYPE==BLDG and non-null address
  - `Buildings_with_age.geojson`
    - Description: list of 44,059 (OBJECTIDs, TYPE, ..., address, geometry, age_category)
    - Note: derived from `Buildings_with_addr.geojson` using `match_building_to_year.py` 
  - `collect_via_gsv_complete.csv`
    - Description: 4,085 buildings from `Buildings.geojson` with TYPE = 'BLDG' that could not be matched to a PID, MBL or address of a property with known built year. When disregarding unit numbers, it contains 4,072 unique addresses.
  - `collect_decisions_final.jsonl`
    - Description: JSONL file listing buildings patch filenames in the format {OBJECTID.jpg} and corresponding predictions (class in 1--4).
  - `worcester_building_age_map.pdf`
    - Description: Map generated using `map_maker.py`

- (NOT AVAILABLE DUE TO Google Street View API RESTRICTIONS) `gsv_scraped_images/`
  - Folder Structure: `{objectid}/`
                      `{objectid}/crops/house/{objectid}[2-9].jpg`
                      `{objectid}/labels/{objectid}.jpg`
                      `final_target_set/{objectid}.jpg`
  - Description: 3,271 images collected from Google Street View using 4,072 unique addresses in `data/collect_via_gsv_complete.csv`.

- `models/`
  - Folder Structure: `cswin/`
                      `simclr/`
  - Description: each subfolder contains the code related to the Deep Learning models in the paper.

- `worcester_gov_site_data_scraper/`
  - `main.py`
    - Description: Collects property records from the Worcester City Government website by attempting PIDs sequentially and save results as a CSV file.
  - `download_images.py`
    - Description: Downloads images using URLs listed in the CSV file collected using the `main.py` script in this folder.

- `bounding_box_crop.ipynb`
  - Description: uses trained YOLOv8s model to extract key patches from building images.

- `gsv_scraper.ipynb`
  - Description: Scrapes images from Google Street View by address.

- `map_maker.py`
  - Decription: Generates a full city-wide age mapping based on a GEOJSON file containing footprints and age data.

- `match_building_to_year.py`
  - Description: Joins building footprints with city data using multiple matching strategies. It has two main purposes:
    1) Get a list of buildings whose images must be retrieves using a street-view API from their addresses.
    2) Join existing building age data and building age predictions with footprint data.

- `plot_confusion_matrix.py`
  - Description: Plots confusion matrices based on the results of the human evaluation.