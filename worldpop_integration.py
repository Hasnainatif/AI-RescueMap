import os
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import urllib.parse

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform
from rasterio.transform import Affine
from rasterio.enums import Resampling

# Defaults so app.py can omit url/path entirely if desired
WORLDPOP_URL_DEFAULT = "https://huggingface.co/datasets/HasnainAtif/worldpop_2024/resolve/main/global_pop_2024_CN_1km_R2025A_UA_v1.tif"
WORLDPOP_DIR_DEFAULT = "data"  # local cache directory


def _filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path) or "worldpop.tif"
    # Ensure we end with .tif
    if not name.lower().endswith(".tif"):
        name += ".tif"
    return name


def _default_path_for_url(url: str | None) -> str:
    os.makedirs(WORLDPOP_DIR_DEFAULT, exist_ok=True)
    if url:
        return os.path.join(WORLDPOP_DIR_DEFAULT, _filename_from_url(url))
    # fallback filename if no url provided
    return os.path.join(WORLDPOP_DIR_DEFAULT, "worldpop_2024_1km.tif")


@st.cache_resource(show_spinner=False)
def download_worldpop_if_needed(url: str, target_path: str) -> str | None:
    """
    Downloads the WorldPop GeoTIFF to target_path if not present.
    Returns the local path or None on failure.
    """
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path

        with requests.get(url, stream=True, timeout=300) as r:  # longer timeout for first download
            r.raise_for_status()
            tmp_path = target_path + ".part"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, target_path)
        return target_path
    except Exception as e:
        st.warning(f"WorldPop download failed: {e}")
        return None


@st.cache_resource(show_spinner=False)
def open_worldpop(url: str | None = None, path: str | None = None):
    """
    Ensures the dataset is present locally and returns a rasterio dataset.
    Cached so it only opens once per session.

    - If 'path' is None, a cache path is derived automatically from 'url'.
    - If 'url' is None, a default URL is used.
    """
    url = url or WORLDPOP_URL_DEFAULT
    path = path or _default_path_for_url(url)

    local = path if os.path.exists(path) else download_worldpop_if_needed(url, path)
    if not local or not os.path.exists(local):
        return None
    try:
        return rasterio.open(local)
    except Exception as e:
        st.warning(f"WorldPop open failed: {e}")
        return None


def latlon_bounds_from_center(center_lat: float, center_lon: float, radius_km: float):
    """
    Returns a lon/lat bounding box expanded by radius_km around center.
    """
    cos_lat = max(0.05, math.cos(math.radians(center_lat)))
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * cos_lat)
    minx = center_lon - deg_lon
    maxx = center_lon + deg_lon
    miny = center_lat - deg_lat
    maxy = center_lat + deg_lat
    return (minx, miny, maxx, maxy)


@st.cache_data(show_spinner=False)
def read_worldpop_window(url: str | None,
                         path: str | None,
                         center_lat: float,
                         center_lon: float,
                         radius_km: float,
                         out_size: tuple[int, int] = (200, 200)) -> pd.DataFrame | None:
    """
    Reads a window from the WorldPop raster around the given center+radius.

    - Only URL is required; 'path' can be None. The file is auto-cached locally.
    - Uses Resampling.average for reading, then scales by the aggregation factor to approximate sum.
    - Returns a DataFrame with columns: lat, lon, population (people per aggregated pixel).
    - Returns None if data not available for the requested window.
    """
    src = open_worldpop(url, path)
    if src is None:
        return None

    # Build lon/lat bounds around center
    bounds_wgs84 = latlon_bounds_from_center(center_lat, center_lon, radius_km)

    # Reproject bounds to the raster CRS if needed
    try:
        if src.crs and src.crs.to_string() != "EPSG:4326":
            minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, *bounds_wgs84, densify_pts=21)
        else:
            minx, miny, maxx, maxy = bounds_wgs84
    except Exception as e:
        st.warning(f"WorldPop bounds transform failed: {e}")
        return None

    # Intersect with dataset bounds
    ds_bounds = src.bounds
    inter_minx = max(minx, ds_bounds.left)
    inter_miny = max(miny, ds_bounds.bottom)
    inter_maxx = min(maxx, ds_bounds.right)
    inter_maxy = min(maxy, ds_bounds.top)

    if inter_minx >= inter_maxx or inter_miny >= inter_maxy:
        # Requested area not covered by dataset
        return None

    try:
        window = from_bounds(inter_minx, inter_miny, inter_maxx, inter_maxy, transform=src.transform)

        # Downsample safely for performance
        out_h, out_w = out_size
        data = src.read(
            1,
            window=window,
            out_shape=(out_h, out_w),
            resampling=Resampling.average  # sum is warp-only; use average for reads
        )

        # Compute transform for the resampled window
        scale_x = window.width / out_w
        scale_y = window.height / out_h
        window_transform = src.window_transform(window)
        out_transform = window_transform * Affine.scale(scale_x, scale_y)

        # Since we used average, multiply by number of source pixels aggregated into each output pixel
        # to approximate a population sum per aggregated pixel.
        data = data.astype(float) * (scale_x * scale_y)

        # Build coordinate grids (x, y in raster CRS)
        rows, cols = np.indices(data.shape)
        xs, ys = rasterio.transform.xy(out_transform, rows, cols, offset="center")
        xs = np.array(xs)  # x/easting or lon
        ys = np.array(ys)  # y/northing or lat

        # Convert to lon/lat if needed
        if src.crs and src.crs.to_string() != "EPSG:4326":
            lon_flat, lat_flat = transform(src.crs, "EPSG:4326", xs.ravel().tolist(), ys.ravel().tolist())
            lon = np.array(lon_flat).reshape(xs.shape)
            lat = np.array(lat_flat).reshape(ys.shape)
        else:
            lon, lat = xs, ys

        arr = np.array(data, dtype=float)

        # Handle nodata / non-finite
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, 0, arr)
        arr = np.where(np.isfinite(arr), arr, 0)
        arr = np.clip(arr, 0, None)

        # Keep only positives for visualization and impact
        mask = arr > 0
        if not np.any(mask):
            return None

        df = pd.DataFrame({
            "lat": lat[mask].ravel(),
            "lon": lon[mask].ravel(),
            "population": arr[mask].ravel()  # approximate people per aggregated pixel
        })

        return df if len(df) > 0 else None

    except Exception as e:
        st.warning(f"WorldPop window read failed: {e}")
        return None
