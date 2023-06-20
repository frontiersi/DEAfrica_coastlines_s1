import datacube
import numpy as np
import xarray as xr
import pandas as pd

from deafrica_tools.datahandling import load_ard, mostcommon_crs
from deafrica_tools.bandindices import calculate_indices
from dea_tools.coastal import model_tides, tidal_tag, pixel_tides, tidal_stats
from deafrica_tools.dask import create_local_dask_cluster
from coastlines.raster import tide_cutoffs,load_tidal_subset
from coastlines.vector import points_on_line, annual_movements, calculate_regressions

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage.filters import threshold_minimum, threshold_otsu
import random

def lee_filter(da, size):
    """
    Apply lee filter of specified window size.
    Adapted from https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python

    """
    img = da.values
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    
    return img_output

def filter_by_tide_height(ds,tide_centre=0.0):
    # per-pixel tide modelling
    ds["tide_m"], tides_lowres = pixel_tides(ds, resample=True)

    # Determine tide cutoff
    tide_cutoff_min, tide_cutoff_max = tide_cutoffs(
        ds, tides_lowres, tide_centre=tide_centre
    )

    # Filter observations using calculated tide cutoffs  
    tide_bool = (ds.tide_m >= tide_cutoff_min) & (
        ds.tide_m <= tide_cutoff_max
    )
    ds_filtered = ds.sel(time=tide_bool.sum(dim=["x", "y"]) > 0)

    # Apply mask, and load in corresponding tide masked data
    ds_filtered = ds_filtered.where(tide_bool)
    return ds_filtered

def query_filter_s1(dc,query,output_crs):
    S1_ascending=load_ard(dc=dc,
                  products=['s1_rtc'],
                  output_crs=output_crs,
                  resampling='bilinear',
                  align=(10, 10),
                  dask_chunks={'time': 1},
                  group_by='solar_day',
                  dtype='native',
                  sat_orbit_state='ascending',
                  **query)
    S1_descending=load_ard(dc=dc,
                  products=['s1_rtc'],
                  output_crs=output_crs,
                  resampling='bilinear',
                  align=(10, 10),
                  dask_chunks={'time': 1},
                  group_by='solar_day',
                  dtype='native',
                  sat_orbit_state='descending',
                  **query)
    S1_ascending["isAscending"] = xr.where(S1_ascending['mask']!=0,1,np.nan)
    S1_ascending["isDescending"] = xr.where(S1_ascending['mask']!=0,0,np.nan)

    S1_descending["isAscending"] = xr.where(S1_descending['mask']!=0,0,np.nan)
    S1_descending["isDescending"] = xr.where(S1_descending['mask']!=0,1,np.nan)

    S1=xr.concat([S1_ascending,S1_descending],dim='time')
    ascending_mask=(S1.isAscending.sum(dim='time')>S1.isDescending.sum(dim='time'))
    descending_mask=(S1.isAscending.sum(dim='time')<=S1.isDescending.sum(dim='time'))
    S1=S1.where((ascending_mask&(S1.isAscending==1))|(descending_mask&(S1.isDescending==1)),np.nan)
    # drop all-nan observations
    S1=S1.dropna(dim='time',how='all')
    return S1

def preprocess_s1(S1_filtered,lee_filtering,filter_size,time_step):
    # The lee filter above doesn't handle null values
    # We therefore set null values to 0 before applying the filter
    valid = xr.ufuncs.isfinite(S1_filtered)
    S1_filtered = S1_filtered.where(valid, 0)

    if lee_filtering==True: # do filtering
        # Create a new entry in dataset corresponding to filtered VV and VH data
        S1_filtered["filtered_vh"] = S1_filtered.vh.groupby("time").apply(lee_filter, size=filter_size)
        S1_filtered["filtered_vv"] = S1_filtered.vv.groupby("time").apply(lee_filter, size=filter_size)
    else: # don't do filtering
        S1_filtered["filtered_vh"]=S1_filtered["vh"]
        S1_filtered["filtered_vv"]=S1_filtered["vv"]

    # Null pixels should remain null
    S1_filtered['filtered_vh'] = S1_filtered.filtered_vh.where(S1_filtered.filtered_vh!=0,np.nan)
    S1_filtered['filtered_vv'] = S1_filtered.filtered_vv.where(S1_filtered.filtered_vv!=0,np.nan)

    # covert to db
    S1_filtered['filtered_vh']=10 * xr.ufuncs.log10(S1_filtered.filtered_vh)
    S1_filtered['filtered_vv']=10 * xr.ufuncs.log10(S1_filtered.filtered_vv)

    S1_filtered['vv_a_vh']=S1_filtered['filtered_vv']+S1_filtered['filtered_vh']
    S1_filtered['vv_m_vh']=S1_filtered['filtered_vv']-S1_filtered['filtered_vh']
    S1_filtered['vv_d_vh']=S1_filtered['filtered_vv']/S1_filtered['filtered_vh']
    
    # annual composites
    # median
    ds_summaries_s1 = (S1_filtered[['filtered_vh','filtered_vv','vv_a_vh','vv_m_vh','vv_d_vh','area']]
                         .resample(time=time_step)
                         .median('time')
                         .compute()
                        )
    # std
    ds_std_s1 = (S1_filtered[['filtered_vh','filtered_vv','vv_a_vh','vv_m_vh','vv_d_vh','area']]
                         .resample(time=time_step)
                         .std('time')
                         .compute()
                        )
    ds_std_s1=ds_std_s1.rename({var:var+'_std' for var in list(ds_std_s1.data_vars)})
    # merge median and std
    ds_summaries_s1=xr.merge([ds_summaries_s1,ds_std_s1])
    return ds_summaries_s1

