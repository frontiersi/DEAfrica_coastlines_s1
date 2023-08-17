import datacube
import numpy as np
import xarray as xr
import pandas as pd

from deafrica_tools.datahandling import load_ard, mostcommon_crs
from deafrica_tools.bandindices import calculate_indices
from dea_tools.coastal import model_tides, tidal_tag, pixel_tides, tidal_stats
from deafrica_tools.dask import create_local_dask_cluster
from coastlines.raster import tide_cutoffs

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage.filters import threshold_minimum, threshold_otsu
from skimage.morphology import binary_erosion,binary_dilation,disk
import random
from sklearn.model_selection import train_test_split

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

def load_s1_by_orbits(dc,query):
    '''
    Function to query and load ascending and descending Sentinel-1 data 
    and add a variable to denote acquisition orbits
    
    Parameters:
    dc: connected datacube
    query: a query dictionary to define spatial extent, measurements, time range and spatial resolution
    
    Returns:
    Queried dataset with variable 'is_ascending' added to denote orbit path
    
    '''
    # load ascending data
    print('\nQuerying and loading Sentinel-1 ascending data...')
    ds_s1_ascending=load_ard(dc=dc,products=['s1_rtc'],resampling='bilinear',align=(10, 10),
                             dtype='native',sat_orbit_state='ascending',**query)
    # add an variable denoting data source
    ds_s1_ascending['is_ascending']=xr.DataArray(np.ones(len(ds_s1_ascending.time)),
                                                 dims=('time'),coords={'time': ds_s1_ascending.time})
    
    # load descending data
    print('\nQuerying and loading Sentinel-1 descending data...')
    ds_s1_descending=load_ard(dc=dc,products=['s1_rtc'],resampling='bilinear',align=(10, 10),
                              dtype='native',sat_orbit_state='descending',**query)
    # add an variable denoting data source
    ds_s1_descending['is_ascending']=xr.DataArray(np.zeros(len(ds_s1_descending.time)),
                                                  dims=('time'),coords={'time': ds_s1_descending.time})
    
    # merge datasets together
    ds_s1=xr.concat([ds_s1_ascending,ds_s1_descending],dim='time').sortby('time')
    
    return ds_s1

def filter_obs_by_orbit(ds_s1):
    '''
    Function to impliment per-pixel filtering of Sentinel-1 observations 
    to keep only observations from the orbit (ascending/descending) with higher frequency over time. 
    
    Each of the Sentinel-1 observations was acquired from either a descending or ascending orbit, 
    which has impacts on the local incidence angle and backscattering value. 
    Here we do the filtering to minimise the effects of inconsistent looking angle and obit direction for each individual pixel.

    Parameters:
    ds_s1: xarray.Dataset
        Time-series observations of Sentinel-1 data, 
        with two required variables: 'is_ascending' denoting orbit path and 'mask' to identify acquisition exent
    
    Returns:
    ds_s1_filtered: xarray.Dataset
        Filtered dataset
    '''

    print('\nFiltering Sentinel-1 product by orbit...')
    cnt_ascending=((ds_s1["is_ascending"]==1)&(ds_s1['mask']!=0)).sum(dim='time')
    cnt_descending=((ds_s1["is_ascending"]==0)&(ds_s1['mask']!=0)).sum(dim='time')
    
    ds_s1_filtered=ds_s1.where(((cnt_ascending>=cnt_descending)&(ds_s1["is_ascending"]==1))|
                               ((cnt_ascending<cnt_descending)&(ds_s1["is_ascending"]==0)))
    # remove intermediate variable
    ds_s1_filtered=ds_s1_filtered.drop_vars(["is_ascending"])
    # drop all-nan time steps
    ds_s1_filtered=ds_s1_filtered.dropna(dim='time',how='all')
    
    return ds_s1_filtered

def process_features_s1(ds_s1,filter_size=None,s1_orbit_filtering=True,time_step='1Y'):
    '''
    Function to implement preprocessing and features generation on Sentinel-1 data
    preprocessing includes speckle filtering (optional), filtering observations by orbit (optional) and conversion to dB
    features include annual median and std of bands/band math
    
    Parameters:
    ds_s1: xarray.Dataset
        Time-series of Sentinel-1 data, with variable 'vh' required
    filter_size: integer or None
        Speckle filtering size
    s1_orbit_filtering: Boolean
        Whether to filter Sentinel-1 observations by orbit
    time_step: string
        time step used to generate temporal composite
        
    Returns:
        xarray.Dataset
        Preprocessed Sentinel-1 data
    '''
    ds_s1_filtered=ds_s1
    
    # apply Lee filtering if required
    if not filter_size is None:
        print('Applying Lee filtering using filtering size of {} pixels...'.format(filter_size))
        # The lee filter above doesn't handle null values
        # We therefore set null values to 0 before applying the filter
        ds_s1_filtered = ds_s1.where(np.isfinite(ds_s1), 0)
        # Create a new entry in dataset corresponding to filtered VV and VH data
        ds_s1_filtered["vh"] = ds_s1_filtered.vh.groupby("time").apply(lee_filter, size=filter_size)
        # Null pixels should remain null, but also including pixels changed to 0 due to the filtering
        ds_s1_filtered['vh'] = ds_s1_filtered.vh.where(ds_s1_filtered.vh!=0,np.nan)
    
    # filter observations by orbit if required
    if s1_orbit_filtering:
        ds_s1_filtered=filter_obs_by_orbit(ds_s1_filtered)

    # Scale to plot data in decibels
    ds_s1_filtered['vh'] = 10 * np.log10(ds_s1_filtered.vh)
    ds_s1_filtered['vv'] = 10 * np.log10(ds_s1_filtered.vv)
    
    # generate features
    print('Calculating features for Sentinel-1')
    ds_s1_filtered['vv_a_vh']=ds_s1_filtered['vv']+ds_s1_filtered['vh']
    ds_s1_filtered['vv_m_vh']=ds_s1_filtered['vv']-ds_s1_filtered['vh']
    
    # annual composites
    print('Generate temporal composites...')
    # median
    ds_summaries_s1 = (ds_s1_filtered[['vh','vv','vv_a_vh','vv_m_vh','area','angle']]
                         .resample(time=time_step)
                         .median('time')
                         .compute()
                        )
    ds_summaries_s1=ds_summaries_s1.rename({var:var+'_med' for var in list(ds_summaries_s1.data_vars)})
    
    # max
    ds_max_s1 = (ds_s1_filtered[['vh','vv','vv_a_vh','vv_m_vh']]
                     .resample(time=time_step)
                     .max('time')
                     .compute()
                    )
    ds_max_s1=ds_max_s1.rename({var:var+'_max' for var in list(ds_max_s1.data_vars)})
    
    # min
    ds_min_s1 = (ds_s1_filtered[['vh','vv','vv_a_vh','vv_m_vh']]
                 .resample(time=time_step)
                 .min('time')
                 .compute()
                )
    ds_min_s1=ds_min_s1.rename({var:var+'_min' for var in list(ds_min_s1.data_vars)})
    
    # std
    ds_std_s1 = (ds_s1_filtered[['vh','vv','vv_a_vh','vv_m_vh']]
                         .resample(time=time_step)
                         .std('time')
                         .compute()
                        )
    ds_std_s1=ds_std_s1.rename({var:var+'_std' for var in list(ds_std_s1.data_vars)})
    
    # merge all datasets
    ds_summaries_s1=xr.merge([ds_summaries_s1,ds_std_s1,ds_max_s1,ds_min_s1])
    
    return ds_summaries_s1

def create_coastal_mask(da,buffer_pixels):
    '''
    Create a simplified coastal zone mask based on time series of Sentinel-2 MNDWI data
    
    Parameters:
    ds_summaries: xarray.DataArray
        Time series of Sentinel-2 MNDWI data
    buffer_pixels: integer
        Number of pixels to buffer coastal zone
    
    Returns:
    coastal_mask: xarray.DataArray 
        A single time buffered coastal zone mask (0: non-coastal and 1: coastal)
    '''
    print('\nCalculating simplified coastal zone mask...')
    # apply thresholding and re-apply nodata values
    nodata = da.isnull()
    thresholded_ds = da>=0
    thresholded_ds = thresholded_ds.where(~nodata)
    # use 20% ~ 80% wet frequency to identify potential coastal zone
    coastal_mask=(thresholded_ds.mean(dim='time') >= 0.2)&(thresholded_ds.mean(dim='time') <= 0.8)
    # buffering
    print('\nApplying buffering of {} Sentinel-2 pixels...'.format(buffer_pixels))
    coastal_mask=xr.apply_ufunc(binary_dilation,coastal_mask.compute(),disk(buffer_pixels))
    return coastal_mask

def collect_training_samples(ds_summaries_s2,ds_summaries_s1,coastal_mask,max_samples):
    '''
    Collect training samples of land and water from Sentinel-1 data at coastal region using classified Sentinel-2 MNDWI
    Parameters: 
    ds_summaries_s2: xarray.Dataset
        Temporarily aggregated Sentinel-2 data
    ds_summaries_s1: xarray.Dataset
        Temporarily aggregated Sentinel-1 features
    coastal_mask:xarray.DataArray 
        A single time buffered coastal zone mask (0: non-coastal and 1: coastal)
    max_samples: integer
        Maximum number of training samples to collect for land/water class
    
    Returns:
        data,labels: Numpy arrays of Sentinel-1 features and land/water class labels
        ds_summaries_s2['iswater']: xarray.DataArray of land/water classification masked to coastal region
    '''
    print('\nCollecting water/non-water samples...')
    # classs: water (1) and land (0)
    ds_summaries_s2['iswater']=xr.where((ds_summaries_s2['MNDWI']>=0)&(coastal_mask==1),1,
                                      xr.where((ds_summaries_s2['MNDWI']<0)&(coastal_mask==1),0,np.nan))
    ds_summaries_s1=ds_summaries_s1.where(~ds_summaries_s2['iswater'].isnull(),np.nan)
    
    # reshape arrays for input to sklearn
    s1_data=ds_summaries_s1.to_array(dim='variable').transpose('x','y','time', 'variable').values
    data_shape = s1_data.shape
    data=s1_data.reshape(data_shape[0]*data_shape[1]*data_shape[2],data_shape[3])
    labels=ds_summaries_s2.iswater.transpose('x','y','time').values.reshape(data_shape[0]*data_shape[1]*data_shape[2],)
    
    # remove NaNs and Infinities
    labels=labels[np.isfinite(data).all(axis=1)]
    data=data[np.isfinite(data).all(axis=1)]
    print('Number of samples available: ',data.shape[0])
    
    # random sampling maximum number of samples per location
    n_samples=np.min([int(max_samples/2),np.sum(labels==1),np.sum(labels==0)])
    ind_water=random.sample(sorted(np.where(labels==0)[0]), n_samples)
    ind_land=random.sample(sorted(np.where(labels==1)[0]), n_samples)
    rand_indices=ind_water+ind_land
    labels=labels[rand_indices]
    data=data[rand_indices]
        
    return data,labels,ds_summaries_s2['iswater']