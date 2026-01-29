import glob
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import zipfile
#from sklearn.model_selection import train_test_split

from utils.utils import *
from utils.cdse_utils import *
from utils.torch import define_model, load_model_weights
from utils.plot import *

import traceback
import sys

# Show DeltaTwin logs
WORKDIR = os.environ.get("DELTA_WORKDIR", "/tmp")
LOG_FILE = os.path.join(WORKDIR, "flood_detector.log")

# Redirect stdout/stderr to log file (optional, but recommended)
sys.stdout = open(LOG_FILE, "a", buffering=1)
sys.stderr = open(LOG_FILE, "a", buffering=1)

def main():

    # Setup
    args = parser.parse_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, "cfg", "config.yaml")
    config = load_config(config_path=config_path)
    DATASET_DIR = os.path.join(BASE_DIR, config['dataset_version'])
    OUTPUT_DIR = os.getenv("DELTA_OUTPUT_DIR", "outputs")
    today = datetime.utcnow().date()

# Create Dataset
    # Parameters from dataset config
    logger.info("Starting dataset creation...")
    query_config = config['query']
    bands = config['bands']
    bbox = args.bbox
    start_date = today if args.start_date.lower() == "today" else datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = (today + timedelta(days=1)) if args.end_date.lower() == "today" else datetime.strptime(args.end_date, "%Y-%m-%d")
    mid_date = start_date + (end_date - start_date) / 2
    max_items = query_config['max_items']
    max_cloud_cover = query_config['max_cloud_cover']

    print(f"Querying data from {start_date} to {end_date} for bbox {bbox}...")

    all_l2a_results = query_sentinel_data(
        bbox, start_date, end_date, max_items, max_cloud_cover
    )

    # Process and align data
    df_l2a = queries_curation(all_l2a_results)
    #df_l2a.to_csv(f"{DATASET_DIR}/output_l2a.csv")

    logger.info("Starting download process...")

    download_sentinel_data(
        df_output = df_l2a,
        base_dir = DATASET_DIR,
        access_key = args.cdse_key,
        secret_key = args.cdse_secret,
        endpoint_url = 'https://eodata.dataspace.copernicus.eu'
    )

    logger.success("All downloads completed.")


# Patchify
    logger.info("Extracting patch coordinates...")
    zarr_dir = os.path.join(DATASET_DIR, "target")
    zarr_files = glob.glob(os.path.join(zarr_dir, "*.zarr"))
    logger.info(f"Zarr files successfully created: {len(zarr_files)}")

    if not zarr_files:
        logger.warning(f"No Zarr files found in {zarr_dir}")
        return

    patches_per_zarr, df_coords = create_patches_dataframe(
        zarr_files,
        bands=bands,
        bbox=bbox,
        target_res='r10m',
        stride=256,
        patch_size=config['download'].get('patch_size', 256),
        date=mid_date,
        pat=args.earth_data_hub_pat
    )
    #np.save(os.path.join(DATASET_DIR, 'patches.npy'), patches)
    logger.success("Patch extraction completed.")

    # visualize_patches_on_tile(
    #     zarr_file=zarr_files[0],
    #     patches_coords=df_coords[df_coords["zarr_file"] == zarr_files[0]],
    #     patch_size=256,
    #     bbox=config["query"]["bbox"],
    #     save_dir=os.path.join(BASE_DIR, "results")
    # )

# Segmentation

    df_coords = df_coords.rename(columns={"x_pix": "x", "y_pix": "y"})

    # Group by tile
    unique_zarrs = df_coords['zarr_file'].unique()

    tif_paths = []

    for zarr_path in unique_zarrs:

        patches = patches_per_zarr[zarr_path] 
        flood_masks = []

        for i, patch in enumerate(patches):

            b3_idx = bands.index("b03") # green
            b8_idx = bands.index("b08") # nir
            b11_idx = bands.index("b11") # swir

            green = patch[:, :, b3_idx].astype(np.float32)
            nir = patch[:, :, b8_idx].astype(np.float32)
            swir = patch[:, :, b11_idx].astype(np.float32)

            eps = 1e-6
            mndwi = (green - swir) / (green + swir + eps)
            ndwi = (green - nir) / (green + nir + eps)

            flood_mask = (ndwi > 0.1).astype(np.uint8)
            flood_masks.append(flood_mask)
            

        binary_mask = stitch_predictions(
            zarr_file=zarr_path,
            df_coords=df_coords,
            flood_masks=flood_masks,
            patch_size=256,
            vote_threshold=0.5
        )

        #output_dir = os.path.join(DATASET_DIR, "outputs")
        # os.makedirs(OUTPUT_DIR, exist_ok=True)
        # output_dir = OUTPUT_DIR

        tif_path = export_geotiff_and_vector(
            zarr_path=zarr_path,
            binary_mask=binary_mask,
            confidence=None,   # or your own metric
            out_dir=BASE_DIR
        )

        crop_tiff_to_bbox(tif_path, args.bbox, tif_path)
        tif_paths.append(tif_path)

    # # Create ZIP of all TIFs
    # zip_path = os.path.join(BASE_DIR, "flood_masks.zip")
    # with zipfile.ZipFile(zip_path, 'w') as zipf:
    #     for tif in tif_paths:
    #         zipf.write(tif, os.path.basename(tif))
    #         os.remove(tif)

        # visualize_final_panel(
        #     zarr_path=zarr_path,
        #     avg_prob=avg_prob,
        #     binary_mask=binary_mask,
        #     df_coords=df_coords.assign(patch_size=256),
        #     out_path=os.path.join("results", "final_panel.png")
        # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single patch")
    parser.add_argument("--cdse_key", type=str, required=True)
    parser.add_argument("--cdse_secret", type=str, required=True)
    parser.add_argument("--earth_data_hub_pat", type=str, required=True)
    #parser.add_argument("--bbox", type=str, help="Bounding box [minx miny maxx maxy]")
    parser.add_argument("--bbox", type=float, nargs=4, help="Bounding box [minx miny maxx maxy]")
    parser.add_argument("--start_date", type=str, required=False, help="Start date in 'YYYY-MM-DD'. 'today' if current date.")
    parser.add_argument("--end_date", type=str, required=False, help="End date in 'YYYY-MM-DD'. 'today' if current date.")
    try:
    # Entire main logic here
        main()  # whatever is in your main()
    except Exception as e:
        with open(LOG_FILE, "w") as f:
            f.write("Exception:\n")
            traceback.print_exc(file=f)
        raise 