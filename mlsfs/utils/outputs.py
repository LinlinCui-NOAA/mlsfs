import numpy as np
import xarray as xr

class WriteNetcdf:
    def __init__(
        self,
        grid_file=None,
        ):
        self.atm_vars = {
            '10u': {
                'index': [0], 
                'attrs': {'long_name': 'u-component of Wind', 'units': 'm s**-1'},
            },
            '10v': {
                'index': [1],
                'attrs': {'long_name': 'y_wind', 'units': 'm s**-1'},
            },
            '10ws': {
                'index': None,
                'attrs': {'long_name': '10m_wind_speed', 'units': 'm s**-1'},
            },
            't2m': {
                'index': [2], 
                'attrs': {'long_name': 'air_temperature', 'units': 'K'},
            },
            'prmsl': {
                'index': [3], 
                'attrs': {'long_name': 'air_pressure_at_sea_level', 'units': 'Pa'},
            },
            'pwat': {
                'index': [4],
                'attrs': {'long_name': 'precipitation_amount', 'units': 'kg m**-2'},
            },
            'u': {
                'index': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                'attrs': {'long_name': 'x_wind', 'units': 'm s**-1'},
            },
            'v': {
                'index': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                'attrs': {'long_name': 'y_wind', 'units': 'm s**-1'},
            },
            'w': {
                'index': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40,41, 42, 43], 
                'attrs': {'long_name': 'w_wind', 'units': 'm s**-1'},
            },
            't': {
                'index': [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
                'attrs': {'long_name': 'temperature', 'units': 'K'},
            },
            'q': {
                'index': [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                'attrs': {'long_name': 'specific_humidity', 'units': '%'},
            },
            'gh': {
                'index': [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
                'attrs': {'long_name': 'geopotential height', 'units': 'dam'},
            },
        }

        self.ocn_vars = {
            'ssh': {
                'index': [0],
                'attrs': {'long_name': 'sea surface height', 'units': 'm'},
            },
            'sst': {
                'index': [1],
                'attrs': {'long_name': 'sea surface temperature', 'units': 'deg C'},
            },
            'temp': {
                'index': [2, 3, 4, 5, 6, 7],
                'attrs': {'long_name': 'potential temperature', 'units': 'K'},
            },
        }
        latlon = np.load(grid_file)

        self.lons = latlon['lon']
        self.lats = latlon['lat']
        self.atm_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        self.ocn_levels = [0.5, 9.8, 47.3, 97.2, 200.3, 301.7]


    def tonetcdf(self, forecast_time, forecast, model_type, outdir, mask=None):

        ds_list = []
        if model_type == 'atm':
            self.levels = self.atm_levels
            for k, v in self.atm_vars.items():
                ds_list.append(self.get_dataxarray(forecast_time, forecast, k, v, mask))

            ds_merged = xr.merge(ds_list)
            ds_merged.to_netcdf(f'{outdir}/coupled_forecast_atm_{forecast_time.strftime("%Y%m%d%H")}.nc')

        elif model_type == 'ocn':
            self.levels = self.ocn_levels
            for k, v in self.ocn_vars.items():
                ds_list.append(self.get_dataxarray(forecast_time, forecast, k, v, mask))
            ds_merged = xr.merge(ds_list)
            ds_merged.to_netcdf(f'{outdir}/coupled_forecast_ocn_{forecast_time.strftime("%Y%m%d%H")}.nc')
        else:
            raise ValueError(f'model type {model_type} is not supported!')

       

    def get_dataxarray(self, time, forecast, var_name, v, mask=None):

        if v['index'] is not None:
            if len(forecast.shape) == 4:
                data = np.squeeze(forecast[0,v['index']])
            elif len(forecast.shape) == 3:
                data = forecast[v['index']]
            else:
                raise ValueError(f'Dimension of forecast either be 3 or 4!')
        else:
            data = np.sqrt(forecast[0,0]**2 + forecast[0,1]**2) # 10m wind speed


        if v['index'] is not None:
            if len(v['index']) == 1:

                if mask is not None:
                    data = np.ma.masked_where(mask==1, data)

                data = np.expand_dims(data, axis=(0))
                da = xr.DataArray(
                    data.astype('float32'),
                    coords={
                        'longitude': self.lons,
                        'latitude': self.lats,
                        'time': np.array([time]),
                    },
                    dims=['time', 'latitude', 'longitude'],
                    name=var_name
                )

            elif len(v['index']) > 1:
                if mask is not None:
                    mask3d = np.broadcast_to(mask == 1, data.shape)
                    data = np.ma.masked_array(data, mask=mask3d)
                data = np.expand_dims(data, axis=(0))
                da = xr.DataArray(
                    data.astype('float32'),
                    coords={
                        'longitude': self.lons,
                        'latitude': self.lats,
                        'level': self.levels,
                        'time': np.array([time]),
                    },
                    dims=['time', 'level', 'latitude', 'longitude'],
                    name=var_name
                )
        else:
            if mask is not None:
                data = np.ma.masked_where(mask==1, data)

            data = np.expand_dims(data, axis=(0))
            da = xr.DataArray(
                data.astype('float32'),
                coords={
                    'longitude': self.lons,
                    'latitude': self.lats,
                    'time': np.array([time]),
                },
                dims=['time', 'latitude', 'longitude'],
                name=var_name
            )

        return da
