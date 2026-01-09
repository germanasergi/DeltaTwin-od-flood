import cv2
import io
import os
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from loguru import logger
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd
import requests
import numpy as np
import time


def extract_s3_path_from_url(url):
    """
    Extracts the S3 object path from an S3 URL or URI.

    This function parses S3 URLs/URIs and returns just the object path portion,
    removing the protocol (s3://), bucket name, and any leading slashes.

    Args:
        url (str): The full S3 URI (e.g., 's3://eodata/path/to/file.jp2')

    Returns:
        str: The S3 object path (without protocol, bucket name and leading slashes)
    """
    # If it's not an S3 URI, return it unchanged
    if not url.startswith('s3://'):
        return url

    # Parse the S3 URI
    parsed_url = urlparse(url)

    # Ensure this is an S3 URL
    if parsed_url.scheme != 's3':
        raise ValueError(f"URL {url} is not an S3 URL")

    # Extract the path without leading slashes
    object_path = parsed_url.path.lstrip('/')

    return object_path


def get_product(s3_resource, bucket_name, object_url, output_path):
    """
    Download a product from S3 bucket and create output directory if it doesn't exist.

    Args:
        s3_resource: boto3 S3 resource object
        bucket_name (str): Name of the S3 bucket
        object_url (str): Path to the object within the bucket
        output_path (str): Local directory to save the file

    Returns:
        str: Path to the downloaded file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Extract filename from the object URL
    _, filename = os.path.split(object_url)

    # Full path where the file will be saved
    local_file_path = os.path.join(output_path, filename)

    print(f"Downloading {object_url} to {local_file_path}...")

    try:
        # Download the file from S3
        s3_resource.Bucket(bucket_name).download_file(object_url, local_file_path)
        print(f"Successfully downloaded to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        raise

    return local_file_path


def remove_last_segment_rsplit(sentinel_id):
    # Split from the right side, max 1 split
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]


def query_sentinel_data(bbox, start_date, end_date, max_items, max_cloud_cover, num_days=10,  max_lookback_days=5):
    """Query Sentinel data for the specified parameters"""
    # Generate the polygon string from bbox [minx, miny, maxx, maxy]
    polygon = f"POLYGON (({bbox[0]} {bbox[1]}, {bbox[0]} {bbox[3]}, {bbox[2]} {bbox[3]}, {bbox[2]} {bbox[1]}, {bbox[0]} {bbox[1]}))"

    # Initialize empty lists to store all results
    all_l2a_results = []

    # Loop through the date range with a step of 10 days
    current_date = start_date
    while current_date < end_date:
        # Calculate the end of the current interval
        next_date = min(current_date + timedelta(days=num_days), end_date)

        # Format the dates for the OData query
        start_interval = f"{current_date.strftime('%Y-%m-%dT00:00:00.000Z')}"
        end_interval = f"{next_date.strftime('%Y-%m-%dT23:59:59.999Z')}"

        date_interval = f"{current_date.strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}"

        try:
            # Query for L2A products
            l2a_query_url = create_cdse_query_url(
                product_type="MSIL2A",
                polygon=polygon,
                start_interval=start_interval,
                end_interval=end_interval,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
                orderby="ContentDate/Start"
            )
            l2a_json = requests.get(l2a_query_url).json()
            l2a_results = l2a_json.get('value', [])

            if not l2a_results:
                # If no data, look back in time
                lookback_days = 1
                found_data = False
                while lookback_days <= max_lookback_days:
                    new_start = current_date - timedelta(days=lookback_days)
                    new_end = next_date - timedelta(days=lookback_days)
                    start_interval = f"{new_start.strftime('%Y-%m-%dT00:00:00.000Z')}"
                    end_interval = f"{new_end.strftime('%Y-%m-%dT23:59:59.999Z')}"
                    date_interval = f"{new_start.strftime('%Y-%m-%d')}/{new_end.strftime('%Y-%m-%d')}"

                    l2a_results = requests.get(
                        create_cdse_query_url(
                            product_type="MSIL2A",
                            polygon=polygon,
                            start_interval=start_interval,
                            end_interval=end_interval,
                            max_cloud_cover=max_cloud_cover,
                            max_items=max_items,
                            orderby="ContentDate/Start"
                        )
                    ).json().get("value", [])

                    if l2a_results:
                        found_data = True
                        logger.info(f"No data for original interval {current_date}/{next_date}, using closest available {date_interval}")
                        break
                    lookback_days += 1

                if not found_data:
                    logger.warning(f"No Sentinel data found within {max_lookback_days} days before {current_date}")
                    current_date = next_date
                    continue  # skip this interval

            # Add interval metadata
            for item in l2a_results:
                item['query_interval'] = date_interval

            l2a_count = len(l2a_results)

            # Append results
            all_l2a_results.extend(l2a_results)

            logger.info(f"L2A Items for {date_interval}: {l2a_count}")
            logger.info("####")

        except Exception as e:
            logger.error(f"Error processing interval {date_interval}: {str(e)}")

        # Move to the next interval
        current_date = next_date

    return all_l2a_results

def queries_curation(all_l2a_results):
    """Process and align L1C and L2A data to ensure they match"""
    # Create DataFrames
    df_l2a = pd.DataFrame(all_l2a_results)

    # Select required columns
    df_l2a = df_l2a[["Name", "S3Path", "Footprint", "GeoFootprint", "Attributes"]]

    # Extract cloud cover
    df_l2a['cloud_cover'] = df_l2a['Attributes'].apply(lambda x: x[2]["Value"])
    # Drop the Attributes column (note: inplace=True needed or need to reassign)
    df_l2a = df_l2a.drop(columns=['Attributes'], axis=1)
    # Create id_key for matching
    df_l2a['id_key'] = df_l2a['Name'].apply(remove_last_segment_rsplit)
    df_l2a['id_key'] = df_l2a['id_key'].str.replace('MSIL2A_', 'MSIL1C_')

    # Remove duplicates
    df_l2a = df_l2a.drop_duplicates(subset='id_key', keep='first')

    return df_l2a



def create_cdse_query_url(
    collection_name="SENTINEL-2",
    product_type="MSIL2A",
    polygon=None,
    start_interval=None,
    end_interval=None,
    max_cloud_cover=100,
    max_items=1000,
    additional_filters=None,
    orderby="ContentDate/Start"  # Add orderby parameter with default value
    ):
    """
    Create a query URL for the Copernicus Data Space Ecosystem OData API.

    Parameters:
    -----------
    collection_name : str
        The collection name (e.g., 'SENTINEL-2', 'SENTINEL-1')
    product_type : str
        The product type (e.g., 'MSIL2A', 'MSIL1C', 'GRD')
    polygon : str
        WKT polygon string for spatial filtering
    start_interval : str
        Start time in ISO format with Z for UTC (e.g., '2023-01-01T00:00:00.000Z')
    end_interval : str
        End time in ISO format with Z for UTC (e.g., '2023-01-31T23:59:59.999Z')
    max_cloud_cover : int
        Maximum cloud cover percentage (0-100)
    max_items : int
        Maximum number of items to return
    additional_filters : list
        List of additional filter strings to add to the query
    orderby : str or None
        Field to order results by (e.g., 'ContentDate/Start', 'ContentDate/Start desc')
        Set to None to skip ordering

    Returns:
    --------
    str
        Complete URL for the OData API query
    """

    # Basic filter for collection
    filter_parts = [f"Collection/Name eq '{collection_name}'"]

    # Add spatial filter if provided
    if polygon:
        filter_parts.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')")

    # Add product type filter
    if product_type:
        filter_parts.append(f"contains(Name,'{product_type}')")

    # Add temporal filters if provided
    if start_interval:
        filter_parts.append(f"ContentDate/Start gt {start_interval}")
    if end_interval:
        filter_parts.append(f"ContentDate/Start lt {end_interval}")

    # Add cloud cover filter if applicable
    # Only add for optical sensors (Sentinel-2)
    if collection_name == 'SENTINEL-2' and max_cloud_cover < 100:
        filter_parts.append(
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
            f"att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
        )

    # Add any additional filters
    if additional_filters:
        filter_parts.extend(additional_filters)

    # Construct the URL with all filters
    filter_string = " and ".join(filter_parts)
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter={filter_string}"

    # Add orderby parameter if specified
    if orderby:
        url += f"&$orderby={orderby}"

    # Add top parameter for limiting results
    url += f"&$top={max_items}"

    url += "&$expand=Attributes"

    return url


def parse_safe_manifest(content: str):
    """
    Parse a Sentinel SAFE manifest file and extract href attributes.

    Args:
        manifest_path (str): Path to the manifest.safe file

    Returns:
        pd.DataFrame: DataFrame containing href values and file information,
                     or None if an error occurred
    """
    try:

        # Parse the content
        root = ET.fromstring(content)

        # Extract all elements with an href attribute using a generic approach
        hrefs = []
        for elem in root.findall(".//*[@href]"):
            href = elem.get('href')
            if href:
                hrefs.append(href)

        # Create DataFrame with href values and file information
        df_files = pd.DataFrame({
            'href': hrefs,
            'file_type': [href.split('.')[-1] if '.' in href else 'unknown' for href in hrefs],
            'file_name': [os.path.basename(href) for href in hrefs]
        })


        return df_files

    except ET.ParseError as e:
        logger.error(f"XML parsing error : {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing manifest: {str(e)}")
        return None


def download_manifest(s3_client, bucket_name, s3_path, max_attempts=5, retry_delay=2):
    """
    Download and parse a Sentinel-2 product manifest file from S3.

    Args:
        s3_client: Boto3 S3 client
        bucket_name (str): S3 bucket name
        s3_path (str): Base S3 path to the product
        max_attempts (int): Maximum number of download attempts
        retry_delay (int): Seconds to wait between retry attempts

    Returns:
        tuple: (success (bool), dataframe of files or None)
    """
    # Extract base S3 URL and create manifest URL
    s3_base_url = extract_s3_path_from_url(s3_path).replace("/eodata", "")
    s3_manifest_url = f"{s3_base_url}/manifest.safe"

    # Try to download manifest file with retry logic
    attempt = 0
    content = None

    while attempt < max_attempts:
        try:
            # Get the manifest file
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_manifest_url)

            # Check if successful
            if response["ResponseMetadata"]['HTTPStatusCode'] == 200:
                content = response['Body'].read()
                logger.info(f"Downloaded manifest from {s3_manifest_url}")
                break
            else:
                logger.warning(f"Unexpected status: {response['ResponseMetadata']['HTTPStatusCode']}")
                attempt += 1
                time.sleep(retry_delay)

        except Exception as e:
            logger.warning(f"Error downloading manifest: {str(e)}")
            attempt += 1
            time.sleep(retry_delay)

    if content is None:
        logger.error(f"Failed to download manifest after {max_attempts} attempts")
        return False, None

    # Parse the manifest into a dataframe
    df_files = parse_safe_manifest(content=content)

    return df_files


def filter_band_files(df_files, bands=None, product_type=None, resolution=None):
    """
    Filter a dataframe for Sentinel-2 band files supporting both L1C and L2A formats.

    Args:
        df_files (pd.DataFrame): DataFrame with 'href' column containing file paths
        bands (list, optional): List of band names to filter for (e.g., ['B02', 'B03', 'B04']).
                               If None, defaults to RGB bands.
        product_type (str, optional): Product type ('L1C' or 'L2A'). If None, both types are included.
        resolution (str or int, optional): Specific resolution to filter for L2A products ('10m', '20m', '60m' or 10, 20, 60).
                                         If None, includes all resolutions.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only requested band files
    """
    # Define default bands to filter if not specified
    if bands is None:
        bands = ['B02', 'B03', 'B04']  # RGB bands by default

    # Convert resolution to string if it's an integer
    if resolution is not None:
        resolution = str(resolution)

    # Build regex patterns to match both L1C and L2A formats
    band_patterns = []

    for band in bands:
        # L1C format: IMG_DATA/*_B02.jp2
        if product_type is None or product_type.upper() == 'L1C':
            band_patterns.append(r'IMG_DATA/.*_' + band + r'\.jp2')

        # L2A formats with correct pattern: IMG_DATA/R20m/T55KGR_20200103T001101_B02_20m.jp2
        if product_type is None or product_type.upper() == 'L2A':
            if resolution:
                # If specific resolution is provided, filter for that resolution
                band_patterns.append(r'IMG_DATA/R' + resolution + r'm/.*_' + band + r'_' + resolution + r'm\.jp2')
            else:
                # If no resolution is specified, include all resolutions
                band_patterns.extend([
                    r'IMG_DATA/R10m/.*_' + band + r'_10m\.jp2',
                    r'IMG_DATA/R20m/.*_' + band + r'_20m\.jp2',
                    r'IMG_DATA/R60m/.*_' + band + r'_60m\.jp2'
                ])

    filter_condition = False
    for pattern in band_patterns:
        filter_condition = filter_condition | df_files['href'].str.contains(pattern, regex=True)

    df_gr = df_files[filter_condition].copy()  # Create a copy to avoid the warning


    # Remove leading ./ from href paths
    df_gr['href'] = df_gr['href'].str.replace(r'^\./', '', regex=True)

    return df_gr


def get_product(s3_client, bucket_name, object_url, output_path,
                           resize=False, target_size=1830, interpolation=cv2.INTER_CUBIC,
                           format='PNG'):
    """
    Retrieve a satellite image from S3, optionally resize it, and save it.

    Parameters:
    -----------
    s3_client : boto3.client
        Initialized S3 client
    bucket_name : str
        S3 bucket name
    object_url : str
        S3 object key/URL for the image
    output_path : str
        Path where the image should be saved
    resize : bool
        Whether to resize the image (default: False)
    target_size : tuple or None
        Target size as (width, height). Required if resize=True.
    interpolation : int
        OpenCV interpolation method (default: cv2.INTER_CUBIC)
    format : str
        Output image format (default: 'JPEG')

    Returns:
    --------
    str : Path to the saved image
    """
    # Get the image content from S3
    product_content = get_product_content(s3_client=s3_client,
                                          bucket_name=bucket_name,
                                          object_url=object_url)

    # Open as PIL Image
    image = Image.open(io.BytesIO(product_content))

    # Only resize if requested
    if resize:

        if target_size is None:
            raise ValueError("target_size must be specified when resize=True")

        # Convert to numpy array
        image_array = np.array(image)

        # Resize using OpenCV
        resized_array = cv2.resize(image_array, (target_size,target_size) , interpolation=interpolation)

        # Convert back to PIL Image
        image = Image.fromarray(resized_array)

    # Save the image (resized or original) to the specified path
    image.save(output_path, format=format)

    return output_path


def download_bands(s3_client, bucket_name, df, bands, product_type, resolution, resize, resize_target, output_dir, max_attempts=10, retry_delay=10):
    """
    Download Sentinel-2 band files from S3 based on dataframe information.

    Args:
        s3_client: S3 client object
        bucket: S3 bucket object
        bucket_name: Name of the S3 bucket
        df (pd.DataFrame): DataFrame with 'S3Path' column containing S3 paths
        bands (list): List of bands to download
        product_type (str): Product type ('L1C' or 'L2A')
        resolution (int, optional): Resolution in meters. Required for L2A products.
        output_dir (str): Base directory to save files
        max_attempts (int): Maximum number of download attempts
        retry_delay (int): Delay between retry attempts in seconds
    """

    for index, row in df.iterrows():
        # Extract base S3 URL
        s3_base_url = extract_s3_path_from_url(row['S3Path']).replace("/eodata","")
        s3_manifest_url = f"{s3_base_url}/manifest.safe"
        _, filename = os.path.split(s3_manifest_url)

        # Try to download manifest file with retry logic
        attempt = 0
        content = None

        while attempt < max_attempts:
            try:
                # Get the manifest file
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_manifest_url)
                # Check if successful
                if response["ResponseMetadata"]['HTTPStatusCode'] == 200:
                    content = response['Body'].read()
                    logger.info(f"Downloaded manifest from {s3_manifest_url}")
                    break
                else:
                    logger.warning(f"Unexpected status: {response['ResponseMetadata']['HTTPStatusCode']}")
                    attempt += 1
                    time.sleep(retry_delay)

            except Exception as e:
                logger.warning(f"Error downloading manifest: {str(e)}")
                attempt += 1
                time.sleep(retry_delay)

        if content is None:
            logger.error(f"Failed to download manifest after {max_attempts} attempts, skipping this product")
            continue

        df_tmp = parse_safe_manifest(content=content)
        df_bands = filter_band_files(df_tmp, bands=bands, product_type=product_type, resolution=resolution)

        for gr in df_bands['href']:
            # Create full S3 URL for the band file
            band_s3_url = f"{s3_base_url}/{gr}"

            # Extract just the filename from the path
            filename = os.path.basename(gr)

            # Extract product ID for folder structure
            path_safe = s3_base_url.split(os.sep)[7].replace(".SAFE","")
            path_save = os.path.join(output_dir, path_safe)
            os.makedirs(path_save, exist_ok=True)

            # Download the file with retry logic
            attempt = 0

            while attempt < max_attempts:
                try:
                    # Download the band file
                    # bucket.download_file(band_s3_url, f"{path_save}/{filename}")

                    output_path = f"{path_save}/{os.path.splitext(filename)[0]}.png"
                    get_product(s3_client=s3_client, bucket_name=bucket_name,
                                object_url=band_s3_url, output_path=output_path,
                           resize=resize, target_size=resize_target, interpolation=cv2.INTER_CUBIC,
                           format='PNG')
                    logger.info(f"Downloaded {filename} to {path_save}")
                    break
                except Exception as e:
                    logger.warning(f"Error downloading band file {filename}: {str(e)}")
                    attempt += 1
                    if attempt < max_attempts:
                        logger.info(f"Retrying download of {filename}, attempt {attempt+1} of {max_attempts}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to download {filename} after {max_attempts} attempts")


def create_rgb_image(path_save, output_filename="RGB.png"):
    """
    Creates an RGB image from B04 (Red), B03 (Green), B02 (Blue) bands.
    
    Args:
        path_save (str): Directory where the bands are saved.
        output_filename (str): Filename for the RGB image.

    Returns:
        str: Path to the saved RGB image.
    """
    try:
        files = os.listdir(path_save)

        b2_path = os.path.join(path_save, next(f for f in files if '_B02' in f and f.endswith('.png')))
        b3_path = os.path.join(path_save, next(f for f in files if '_B03' in f and f.endswith('.png')))
        b4_path = os.path.join(path_save, next(f for f in files if '_B04' in f and f.endswith('.png')))

        b2 = cv2.imread(b2_path, cv2.IMREAD_UNCHANGED)
        b3 = cv2.imread(b3_path, cv2.IMREAD_UNCHANGED)
        b4 = cv2.imread(b4_path, cv2.IMREAD_UNCHANGED)

        rgb = cv2.merge([b4, b3, b2])

        rgb_path = os.path.join(path_save, output_filename)
        cv2.imwrite(rgb_path, rgb)

        logger.info(f"Created RGB image at: {rgb_path}")
        return rgb_path

    except Exception as e:
        logger.error(f"Error creating RGB at {path_save}: {e}")
        return None