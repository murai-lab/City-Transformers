"""
Building footprint to city data matching pipeline.

This script joins building footprints with city data using multiple matching strategies:
1. Direct MBL (parcel ID) matching
2. Address matching (street number + street name)
3. Range matching (for multi-unit buildings with number ranges like 100-110)
4. Manual collection decisions from GSV imagery

The output is a list of unmatched building addresses for further processing.
"""

import pandas as pd
import geopandas as gpd
import json
import os
import re
import sys
import numpy as np


def parse_range(range_str):
    """Parse a range-like string and return (min, max) integer tuple.

    Handles values like '54-60', '158A-158B', '100', and returns
    (min_num, max_num). Returns None if no numeric parts found.
    """
    if pd.isna(range_str):
        return None
    s = str(range_str).strip()
    if s == '':
        return None

    # split on hyphen-like characters
    parts = re.split(r'[\-–—]', s)
    nums = []
    for p in parts:
        m = re.search(r'(\d+)', p)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                continue

    if not nums:
        return None

    return (min(nums), max(nums))


def categorize_year_built(year_built):
    """Map Year_Built to age categories: <1941: 1, 1941-1970: 2, 1971-1990: 3, >1990: 4"""
    if pd.isna(year_built):
        return None
    elif year_built < 1941:
        return 1
    elif 1941 <= year_built <= 1970:
        return 2
    elif 1971 <= year_built <= 1990:
        return 3
    else:  # year_built > 1990
        return 4


def load_and_preprocess_city_data(city_data_path):
    """Load and preprocess city crawler data."""
    df = pd.read_csv(city_data_path, index_col='PID')
    print(f'Loaded city data: {len(df)} records')
    
    # Rename and add age category
    df.rename(columns={'acctnum': 'MBL'}, inplace=True)
    df['age_category'] = df['Year_Built'].apply(categorize_year_built)
    
    # Normalize Location field
    df['Location'] = df['Location'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['Location'] = df['Location'].str.replace(r'^0+([1-9][0-9]*) ', r'\1', regex=True)
    df['Location'] = df['Location'].str.replace(r'^00+', '0', regex=True)
    df['Location'] = df['Location'].str.replace('UNIT ', '', regex=True)
    df = df[df['Location'].str.match(r'^[0-9]+', na=False)]
    
    # Extract address components
    regex_pattern = r'^(?P<STREET_NUM>\d+)(?:[A-Z]|\s[A-Z])?\s+(?P<STREET_TEXT>[\w\s]+?)\s*(?P<UNIT_NUM>#\s*[\w-]+)?$'
    df[['STREET_NUM', 'STREET_TEXT', 'UNIT_NUM']] = df['Location'].str.extract(regex_pattern, expand=True)
    df['UNIT_NUM'] = df['UNIT_NUM'].str.replace(r'#\s*', '', regex=True).str.strip()
    df['ADDRESS'] = df['STREET_NUM'].astype(str) + ' ' + df['STREET_TEXT']
    
    print(f'City data preprocessed: {len(df)} records, {df["age_category"].notna().sum()} with age_category')
    return df


def load_building_footprints(geojson_path):
    """Load and preprocess building footprints."""
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[(gdf['TYPE']=='BLDG')]
    gdf['STREET_TEXT'] = gdf['STREET_TEXT'].str.strip()
    gdf['STREET_NUM'] = gdf['STREET_NUM'].str.strip()
    print(f'Loaded building footprints: {len(gdf)} records')
    return gdf


def match_by_mbl(buildings_gdf, city_data):
    """Match buildings to city data by MBL (parcel ID)."""
    mbl_set = set(city_data['MBL'].dropna().unique())
    mbl_to_age = city_data.dropna(subset=['age_category']).set_index('MBL')['age_category'].to_dict()
    buildings_gdf['age_category'] = buildings_gdf['MBL'].map(mbl_to_age)
    
    matched_count = buildings_gdf['age_category'].notna().sum()
    print(f'Matched by MBL: {matched_count} records')
    return mbl_set


def match_by_address(buildings_gdf, city_data):
    """Match buildings to city data by direct address match."""
    unmatched_mask = buildings_gdf['age_category'].isna()
    
    address_to_age = city_data.dropna(subset=['age_category']).set_index('ADDRESS')['age_category'].to_dict()
    buildings_gdf.loc[unmatched_mask, 'age_category'] = (
        buildings_gdf.loc[unmatched_mask, 'ADDRESS'].map(address_to_age)
    )
    
    new_matched = buildings_gdf['age_category'].notna().sum()
    print(f'Matched by address: {new_matched} total records')


def match_by_range(buildings_gdf, city_data):
    """Match buildings by street name + number range."""
    unmatched_mask = buildings_gdf['age_category'].isna()
    unmatched_buildings = buildings_gdf[unmatched_mask]
    
    # Group city data by street name
    street_text_groups = city_data.dropna(subset=['age_category']).groupby('STREET_TEXT')
    
    # For each unmatched building, check if its street name has entries in city data
    # and if its street number falls within any range
    idx_to_age = {}
    for idx, building in unmatched_buildings.iterrows():
        street_text = building['STREET_TEXT']
        street_num_range = parse_range(building['STREET_NUM'])
        
        if street_text in street_text_groups.groups and street_num_range is not None:
            group = street_text_groups.get_group(street_text)
            for _, row in group.iterrows():
                try:
                    street_num = int(str(row['STREET_NUM']).strip())
                    if street_num_range[0] <= street_num <= street_num_range[1]:
                        idx_to_age[idx] = row['age_category']
                        break
                except ValueError:
                    continue
    
    buildings_gdf.loc[idx_to_age.keys(), 'age_category'] = list(idx_to_age.values())
    print(f'Matched by range: {len(idx_to_age)} records, total: {buildings_gdf["age_category"].notna().sum()}')


def match_by_collection_decisions(buildings_gdf, decisions_path):
    """Match unmatched buildings using manual collection decisions from GSV imagery."""
    unmatched_mask = buildings_gdf['age_category'].isna()
    unmatched_buildings = buildings_gdf[unmatched_mask]
    
    # Load collection decisions
    decisions = {}
    with open(decisions_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            decisions[int(data['filename'].replace('.jpg', ''))] = int(data['prediction'])
    
    # Group unmatched buildings by street address
    grouped = unmatched_buildings.groupby(['STREET_NUM', 'STREET_TEXT'])['OBJECTID'].apply(list)
    
    # Filter decisions for each group
    filtered_decisions = {}
    for (street_num, street_text), objectids in grouped.items():
        filtered_decisions[(street_num, street_text)] = [
            decisions[objid] for objid in objectids if objid in decisions
        ]
        # Warn if there are conflicting decisions for the same address
        unique_vals = set(filtered_decisions[(street_num, street_text)])
        if len(unique_vals) > 1:
            print(f'⚠️  Multiple decisions for {street_num} {street_text}: {unique_vals}')
    
    # Map decisions back to buildings (use median if conflicting)
    idx_to_age = {}
    for idx, row in unmatched_buildings.iterrows():
        key = (row['STREET_NUM'], row['STREET_TEXT'])
        if key in filtered_decisions and filtered_decisions[key]:
            idx_to_age[idx] = np.median(filtered_decisions[key])
    
    buildings_gdf.loc[idx_to_age.keys(), 'age_category'] = list(idx_to_age.values())
    print(f'Matched by collection decisions: {len(idx_to_age)} records, total: {buildings_gdf["age_category"].notna().sum()}')


def get_unmatched_buildings(buildings_gdf, exclude_collected=False, collected_csv_path=None):
    """Get list of unmatched buildings, optionally excluding those in GSV collection."""
    unmatched_mask = buildings_gdf['age_category'].isna()
    unmatched = buildings_gdf[unmatched_mask].copy()
    
    print(f'Unmatched buildings (before filtering): {len(unmatched)}')
    
    # create ADDRESS column early so we can compare to collected Location values
    unmatched['ADDRESS'] = (
        unmatched['STREET_NUM'].astype(str).str.extract(r'([0-9]+)', expand=False).fillna('')
        + ' '
        + unmatched['STREET_TEXT'].astype(str).str.strip()
    )

    if exclude_collected and collected_csv_path:
        # Load collected addresses
        collected_df = pd.read_csv(collected_csv_path, usecols=['OBJECTID', 'Location'])
        collected_objectids = set(collected_df['OBJECTID'].dropna().tolist())
        collected_locations = set(collected_df['Location'].dropna().astype(str).str.strip().str.upper())

        # Remove buildings that were collected by OBJECTID OR by matching ADDRESS to collected Location
        mask_collected = (
            unmatched['OBJECTID'].isin(collected_objectids) |
            unmatched['ADDRESS'].astype(str).str.strip().str.upper().isin(collected_locations)
        )
        unmatched = unmatched[~mask_collected].copy()
        print(f'After excluding collected (by OBJECTID or ADDRESS): {len(unmatched)}')
    
    # Get unique addresses, drop missing street info
    unmatched = unmatched.dropna(subset=['STREET_NUM', 'STREET_TEXT'])
    unmatched = unmatched.drop_duplicates(subset=['STREET_NUM', 'STREET_TEXT'], keep='first')
    
    # drop rows missing street info and deduplicate by street/num
    unmatched = unmatched.dropna(subset=['STREET_NUM', 'STREET_TEXT'])
    unmatched = unmatched.drop_duplicates(subset=['STREET_NUM', 'STREET_TEXT'], keep='first')
    print(f'Unique unmatched addresses: {len(unmatched)}')
    return unmatched


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Match building footprints to city data.')
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Sub-commands: get-scrape-list or join-age-predictions')

    # get-scrape-list: produce the CSV of addresses to query, excluding already collected entries
    sp_get = subparsers.add_parser('get-scrape-list', help='Generate list of addresses to scrape via GSV')
    sp_get.add_argument('--collected-csv', default='collect_via_gsv_complete.csv',
                        help='CSV of already queried OBJECTIDs (default: collect_via_gsv_complete.csv)')

    # join-age-predictions: join age predictions back into buildings GeoDataFrame
    sp_join = subparsers.add_parser('join-age-predictions', help='Join age predictions from GSV to building footprints')
    sp_join.add_argument('--decisions-jsonl', default='collect_decisions_final.jsonl',
                         help='JSONL file with collected decisions (default: collect_decisions_final.jsonl)')

    args = parser.parse_args()

    # Load data
    city_data = load_and_preprocess_city_data('worcester_crawler/worcester_city_data.csv')
    buildings_gdf = load_building_footprints('Buildings_with_addr.geojson')

    # Load city boundary and filter footprints to those within the city polygon
    try:
        city_boundary = gpd.read_file('map_generation_kit/City_Boundary.geojson')
        # Ensure CRS match before geometric test
        if buildings_gdf.crs != city_boundary.crs:
            city_boundary = city_boundary.to_crs(buildings_gdf.crs)
        city_poly = city_boundary.union_all()
        before_count = len(buildings_gdf)
        buildings_gdf = buildings_gdf[buildings_gdf.geometry.within(city_poly)].copy()
        print(f'Filtered buildings by city boundary: {before_count} -> {len(buildings_gdf)} records')
    except Exception as e:
        print(f'Warning: failed to filter by city boundary: {e}')

    # Matching pipeline (common steps)
    print('\n--- Matching Pipeline ---')
    match_by_mbl(buildings_gdf, city_data)
    match_by_address(buildings_gdf, city_data)
    match_by_range(buildings_gdf, city_data)

    if args.command == 'get-scrape-list':
        # Task 1: generate list of addresses to be scraped via GSV, excluding those in collected CSV
        print('\n--- Generating scrape list (exclude collected) ---')
        collected_csv = getattr(args, 'collected_csv', 'collect_via_gsv_complete.csv')
        unmatched = get_unmatched_buildings(
            buildings_gdf,
            exclude_collected=False,
            collected_csv_path=collected_csv
        )

        unmatched.to_csv(
            'unscraped_gsv_buildings.csv',
            columns=['OBJECTID', 'MBL', 'STREET_NUM', 'STREET_TEXT', 'ADDRESS'],
            index=False
        )
        print(f'✓ Saved {len(unmatched)} addresses to unscraped_gsv_buildings.csv')

    elif args.command == 'join-age-predictions':
        # Task 2: join age predictions back into buildings; use decisions JSONL
        print('\n--- Joining age predictions into buildings GeoDataFrame ---')
        decisions_file = getattr(args, 'decisions_jsonl', 'collect_decisions_final.jsonl')
        if os.path.exists(decisions_file):
            match_by_collection_decisions(buildings_gdf, decisions_file)
        else:
            print(f'Warning: decisions file not found: {decisions_file}; skipping collection-based matching')

        # save the buildings GeoDataFrame with computed `age_category`
        try:
            buildings_gdf.to_file('Buildings_with_age.geojson', driver='GeoJSON')
            print('Saved buildings GeoDataFrame with age_category to Buildings_with_age.geojson')
        except Exception as e:
            print(f'Warning: failed to save Buildings_with_age.geojson: {e}')


if __name__ == '__main__':
    main()
