import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

import gcsfs
fs = gcsfs.GCSFileSystem()
import os
import cftime

def load_dataset_interp(input_vars, output_vars):
    basedir = "saved_data/input"
    dataset = xr.open_dataset(os.path.join(basedir, input_vars[0] + "_saved.nc"))
    for var in input_vars[1:]:
        ds = xr.open_dataset(os.path.join(basedir, var + "_saved.nc"))
        dataset[var] = ds[var]
        
    basedir = "saved_data/output"
    for var in output_vars:
        ds = xr.open_dataset(os.path.join(basedir, var + "_saved.nc"))
        dataset[var] = ds[var]
    return(dataset)

def load_dataset(input_vars, output_vars, downsample=True):
    # raw files, not interpolated according to Yu suggestion
    mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
    inp = xr.open_dataset(mapper, engine='zarr', chunks={})
    mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
    output = xr.open_dataset(mapper, engine='zarr', chunks={})
    if downsample: # might as well do first
        inp = inp.isel(sample = np.arange(36,len(inp.sample),72)) #  every 1 day
        output = output.isel(sample = np.arange(36,len(output.sample),72))
    ds = inp[input_vars]
    for var in output_vars:
        ds['out_'+var] = output[var]

    time = pd.DataFrame({"ymd":inp.ymd, "tod":inp.tod})
    # rename sample to reformatted time column 
    f = lambda ymd, tod : cftime.DatetimeNoLeap(ymd//10000, ymd%10000//100, ymd%10000%100, tod // 3600, tod%3600 // 60)
    time = time.apply(lambda x: f(x.ymd, x.tod), axis=1)
    ds['sample'] = list(time)
    ds = ds.rename({'sample':'time'})
    
    mapper = fs.get_mapper("gs://leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.grid-info.zarr")
    ds_grid = xr.open_dataset(mapper, engine='zarr')
    lat = ds_grid.lat.values.round(2) 
    lon = ds_grid.lon.values.round(2)  

    ds['lat'] = (('ncol'),lat.T) # (('sample', 'ncol'),lat.T)
    ds['lon'] = (('ncol'),lon.T)
    
    ds['lat'] = (('ncol'),lat.T) # (('sample', 'ncol'),lat.T)
    ds['lon'] = (('ncol'),lon.T)

    # set multi-index for the original dataset using lat and lon
    ds = ds.set_index(index_id=["lat", "lon"])
    index_id = ds.index_id
    ds = ds.drop(['index_id', 'lat', 'lon'])
    ds = ds.rename({'ncol':'index_id'})
    ds = ds.assign_coords(index_id = index_id)
    
    
    return(ds)

def split_dataset(ds):
    inp = []
    out = []
    for var in ds.data_vars:
        if(var[:3] == 'out'):
            out.append(var)
        else:
            inp.append(var)
    return(ds[inp], ds[out])
        
        
    
def list_all_vars():
    mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
    ds = xr.open_dataset(mapper, engine='zarr')
    all_input_vars = list(ds.data_vars)[:-2]
    
    mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
    ds = xr.open_dataset(mapper, engine='zarr')
    all_output_vars = list(ds.data_vars)[:-2]
    return(all_input_vars, all_output_vars)

keys = {
    'cam_in_ASDIF' : 'diffuse_sw_albedo',
}



def split_vars(var_list, out=False):
    v = []
    leveled = []
    for var in var_list:
        if out:
            var = 'out_' + var
        if(len(ds[var].shape) > 2):
            leveled.append(var)
        else:
            v.append(var)
    return(v, leveled)