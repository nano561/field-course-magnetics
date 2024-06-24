import pandas as pd
import numpy as np
import xarray as xr


class Magnetics:
    def __init__(self, df):
        # properties
        self.p = dict(
            dLine=0,
            dMark=0,
            dSensors=0,
        )

        # rename columns
        df.columns = [x.lower() for x in df.columns]
        df = df.rename(columns={'reading': 'BL'})
        df = df.reset_index()

        # string to datetime
        d = df['date'] + ' ' + df['time']
        df['time'] = pd.to_datetime(d, format='%m/%d/%y %H:%M:%S.%f')
        df = df.drop(columns=['date'])

        # # separate date
        # self.date = df['time'].dt.date.unique()
        # df['time'] = df['time'].dt.time

        # separte fields
        half = round(len(df)/2)

        d1 = df.iloc[:half].set_index('time')
        d2 = df.iloc[half:].set_index('time')

        # get sensor spacing
        dDiff = abs(d1['x']-d2['x'])
        self.p['dSensors'] = dDiff.unique().item()
        print(f'Sensor spacing: {self.p["dSensors"]:.4f} m')

        d1['BR'] = d2['BL']
        d1['x'] = (d1['x']+d2['x'])/2

        # reorganize
        self.df = d1.reset_index()
        self.df = self.df.sort_values(by=['time'])
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.drop(columns=['index'])

        self.df = self.df[['time', 'line', 'mark', 'BL', 'BR', 'x', 'y']]

        # get direction for bidirectional survey
        self.get_direction()

    def remove_median_field(self):
        """
        Removes the median magnetic field.
        """
        m = self.df[['BR', 'BL']].stack().median()

        c = ['BR', 'BL']
        self.df[c] = self.df[c] - m
        print(f'Mean field: {m:.2f} nT')

        self.p['median_field'] = m

        return self

    def stackFields(self, df=None):
        """
        Stack the two sensor fields into a single column. Go from "sensor midpoint x" to "sensor position x".
        """
        if df is None:
            df = self.df

        dr = df.copy(deep=True)[['x', 'y', 'BR']]
        dl = df.copy(deep=True)[['x', 'y', 'BL']]

        ds = df['direction'] * self.p['dSensors']/2

        dr['x'] = dr['x'] + ds
        dl['x'] = dl['x'] - ds

        dr.rename(columns={'BR': 'B'}, inplace=True)
        dl.rename(columns={'BL': 'B'}, inplace=True)

        d = pd.concat([dr, dl])

        return d

    def time2sec(self):
        """
        Convert datetime to seconds.
        """
        # time to seconds
        self.df.sort_values(by=['time'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.p['start_time'] = self.df['time'].min()
        self.df['time'] -= self.p['start_time']

        self.df['time'] = self.df['time'].dt.total_seconds()
        self.df.rename(columns={'time': 'sec'}, inplace=True)
        return self

    def quantile(self, q):
        df = self.df[['BR', 'BL']].stack()
        q = df.quantile([q, 1-q]).to_numpy()
        return q

    def estimate_line_spacing(self):
        """
        Estimate the line spacing from available x-coordinates.
        """
        d = self.df['x'].sort_values().diff().value_counts()
        s = d.index[1]

        print(f'Estimated line spacing: {s:.2f} m')

        self.p['dLine'] = s

    def estimate_mark_spacing(self):
        """
        Estimate the mark spacing from available y-coordinates.
        """
        d = self.df.sort_values(by=['line', 'mark'])
        s = d['mark'].diff()
        s = np.sign(s)

        d = d.loc[s == 1, 'y'].diff()
        d = d[d > 0]
        # print(d.mean())
        # print(d.std())

        s = d.mean().round(1)

        print(f'Estimated mark spacing: {s:.2f} m')

        self.p['dMark'] = s

    def get_direction(self):
        """
        Get the direction (up or down) of the bidirectional survey.
        """
        # get direction for bidirectional survey
        for t in ['time', 'sec']:
            if t in self.df.columns:
                break

        d = self.df.sort_values(by=['x', t])
        d.reset_index(drop=True, inplace=True)

        d['dy'] = d['y'].diff()
        s = d.groupby('line')['dy'].median()
        s = np.sign(s)
        self.df['direction'] = d['line'].map(s)

        return self

    def get_direction2(self):
        """
        Get the direction (up or down) of the bidirectional survey.
        """
        dir = self.df['mark'].diff()

        dir = np.sign(dir)
        dir = dir.replace(0, np.nan)
        dir = dir.interpolate(method='pad')
        self.df['direction'] = dir
        return self

    def estimate_sensor_spacing(self):
        """
        Estimate the sensor spacing from available x-coordinates.
        """
        d = self.df.sort_values(by=['line', 'mark', 'sec'])

        a = d['x'].diff().abs().value_counts()

        s = a.index[0]
        print(f'Estimated sensor spacing: {s:.2f} m')
        return s

    def estimate_line_beginning(self):
        """
        Find the sample at the beginning of each line.
        """
        i = self.df['line'].diff() == 1
        i = self.df.index[i]
        a = self.df.loc[i, 'y'].round(0).unique()
        a = np.sort(a)
        return a

    def get_mark_times(self):
        """
        Get the acquisition time for each mark.
        """
        d = self.df
        d = d.reset_index()
        d = d.groupby(['line', 'mark'])
        v = d['sec'].max() - d['sec'].min()

        return v

    def interpolate_marks(self):
        """
        Linearly interpolate between marks.
        Assumes that the walking speed is constant. The higher the deviation from this assumtion the higher the introduced error.
        """

        d = self.df
        dm = d['mark'].diff().abs()
        dm = dm + d['line'].diff().abs()
        # dm[dm > 0] = 1
        i = dm[dm > 0].index

        # interpolate
        d['int'] = np.nan
        d.loc[i-1, 'int'] = 1
        d.loc[i, 'int'] = 0
        d.loc[0, 'int'] = 0
        d.loc[d.index[-1], 'int'] = 1
        d['int'] = d['int'].interpolate(method='linear')
        self.df['int'] = d['int']

        return self

    def separate_fields(self):
        """
        Separate the two sensor fields into separate columns.
        """
        dSensors = self.estimate_sensor_spacing()

        # find fields
        d = self.df.sort_values(by=['line', 'mark', 'time'])
        d['ih'] = d['x'] + dSensors/2
        d['ih'] = d['ih'] / dSensors
        d['ih'] = np.mod(d['ih'], 2)

        f1 = d[d['ih'] == 0]
        f2 = d[d['ih'] == 1]

        f1 = f1.set_index('time')
        f2 = f2.set_index('time')

        f1.drop(columns=['index', 'ih'], inplace=True)
        f2.drop(columns=['index', 'ih'], inplace=True)

        f2 = f2.add_suffix('2')

        d = pd.concat([f1, f2], axis=1)

        # check consistency
        for k in ['y', 'line', 'mark', 'direction']:
            assert all(d[k] == d[k+'2'])
            d.drop(columns=[k+'2'], inplace=True)

        assert all(d['x'] + dSensors == d['x2'])

        d['x'] = d['x'] + dSensors/2
        d.drop(columns=['x2'], inplace=True)

        d.reset_index(inplace=True)

        d = d.sort_values(by=['time', 'line', 'mark'])
        d = d.reset_index()
        o = ['time', 'line', 'mark', 'H', 'H2', 'x', 'y', 'direction']
        self.df = d[o]

        return self

    def line2x(self, dLine=0):
        """
        Convert line number to x-coordinate.
        """
        if dLine == 0:
            dLine = self.estimate_line_spacing()

        self.df['x2'] = self.df['line'] * dLine
        return self

    def mark2y(self, dMark=0):
        """
        Convert mark number to y-coordinate.
        """
        if dMark == 0:
            dMark = self.estimate_mark_spacing()

        self.df.sort_values(by=['line', 'mark'], inplace=True)

        l = self.df['mark'] + (1 - self.df['direction'])/2
        l += self.df['int'] * self.df['direction']

        self.df['y2'] = l * dMark
        return self


def write_esri(d, fname, fext='grd'):
    """
    Write a pandas DataArray to an ESRI ASCII grid file.
    Note that data has to be on a regular grid.
    """

    # check dataformat
    dx = d['E'].diff('E')
    dy = d['N'].diff('N')
    if abs(max(dx) - min(dx)) > 10e-5 and abs(max(dy) - min(dy)) > 10e-5:
        raise Exception(
            'Error writing grid data to ESRI GIRD file.n Grid size is not regular! Quitting...')

    if (dx[0] - dy[0]) > 1e-6:
        raise Exception(
            'Error writing grid data to ESRI GIRD file. Grid size is not rectangular! Quitting...')

    # Prepare entries for text file header
    nrow, ncol = d.shape
    xllcorner = np.min(d['E'])
    yllcorner = np.min(d['N'])
    cellsize = (d['E'][1] - d['E'][0])
    NODATA_value = -9999

    # Reformat data and replace nan with NODATA_value
    a = d.transpose('E', 'N').fillna(NODATA_value).to_numpy()
    a = np.fliplr(a)  # image origin is top left
    a = a.flatten(order='F')  # write in column major order

    # Write to file
    f = f"{fname}.{fext}"
    with open(f, 'w') as fid:
        # Write header
        fid.write(f"ncols         {ncol}\n")
        fid.write(f"nrows         {nrow}\n")
        fid.write(f"xllcorner     {xllcorner:.8f}\n")
        fid.write(f"yllcorner     {yllcorner:.8f}\n")
        fid.write(f"cellsize      {cellsize:.8f}\n")
        fid.write(f"NODATA_value  {NODATA_value}\n")

        # Write data
        s = a.round(3).astype(str)

        # pad to multiple of 10
        s = np.pad(s, (0, 10-len(s) % 10), 'constant', constant_values='')

        # reshape to matrix
        s = np.reshape(s, (-1, 10))

        # add newline character
        s = np.pad(s, ((0, 0), (0, 1)), 'constant', constant_values='\n')

        # remove last entry in last row
        s[-1, -1] = ''

        # write to file
        s = s.flatten()
        s = ' '.join(s)
        fid.write(s)

        # old slow method
        # format_str = ' '.join(['%7.1f'] * 10) + '\n'
        # for i in range(0, len(dstack), 10):
        #     fid.write(format_str % tuple(dstack[i:i+10]))

    print(f"Griddata written to: \n{f}")


def xrRegularRectangular(d):
    """
    Interpolate rectangular grid to quadratic grid.
    """
    dx = d['x'][1]-d['x'][0]
    dy = d['y'][1]-d['y'][0]

    dl = np.min(np.array([dx, dy]))
    l = np.max(np.array([d['x'].max(), d['y'].max()]))+dl

    # interpolate to rectangular grid
    xs = np.arange(0, l, dl)

    d = d.interp(x=xs, y=xs)
    return d


def coordinates2image(d, tCo):
    oversampling = 3
    c = dict()

    index = ['top_left', 'bottom_left', 'bottom_right', 'top_right']

    xmax = d['x'].max().item()
    ymax = d['y'].max().item()
    nx = len(d['x'])-1
    ny = len(d['y'])-1

    c['input'] = np.array(
        [[0, 0], [0, ny], [nx, ny], [nx, 0]], dtype='float32')
    c['input'] = pd.DataFrame(c['input'], columns=['x', 'y'], index=index)

    c['origin'] = dict(
        E=tCo.loc['bottom_left']['E'],
        N=tCo.loc['top_left']['N']
    )

    # image coordinates: origin is top left, y-axis points down
    c['output'] = tCo.copy(deep=True)
    c['output']['E'] -= c['origin']['E']
    c['output']['N'] -= c['origin']['N']
    c['output']['N'] *= -1

    # oversampling
    length = c['output'].max().max()
    n = np.max([nx, ny]) * oversampling
    n = int(np.ceil(n))
    c['axisOut'] = np.linspace(0, length, n)

    # length to pxels
    c['pixelDensity'] = n/length
    c['output'] *= c['pixelDensity']

    # transfrom image coordinates to real coordinates
    c['coordsOut'] = dict(
        E=c['axisOut'] + c['origin']['E'],
        N=-c['axisOut'] + c['origin']['N']
    )

    c['input_pts'] = c['input'].loc[index].to_numpy(dtype='float32')
    c['output_pts'] = c['output'].loc[index].to_numpy(dtype='float32')

    return c


def transform_perspective(d, M, coordsOut):
    import cv2

    img = d.transpose('x', 'y').to_numpy()
    img = np.flip(img.T, axis=0)
    sizeOut = (len(coordsOut['E']),)*2

    out = cv2.warpPerspective(
        img,
        M,
        sizeOut,
        flags=cv2.INTER_NEAREST,
        borderValue=np.nan)

    d = xr.DataArray(
        out.T,
        dims=['E', 'N'],
        coords=coordsOut
    ).transpose('N', 'E')

    return d.sortby('E').sortby('N')


def get_affine_matrix(rotation=0, shear=np.pi/2, translation=[0, 0]):
    # default values

    # shear matrix with conserved (scaled) y-axis
    S = np.array([[1, 1/np.tan(shear)], [0, np.sin(shear)]])

    # rotation matrix
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation), np.cos(rotation)]])

    # transformation matrix
    M = np.eye(3)
    M[0:2, 0:2] = R@S
    M[0:2, 2] = translation
    return M


def transform_affine(d, M, resolution=1000, method='linear'):
    """
    Rotate and the coordinate system of the survey grid to measured coordinates.
    """
    from scipy.interpolate import griddata

    # define data grid
    dstack = d.stack(s=('x', 'y'))
    grid_data = np.array([dstack['x'], dstack['y']]).T
    grid_data = np.einsum('ai,bi->ba', M[0:2, 0:2], grid_data)

    # define interpolation grid
    origin = np.min(grid_data, axis=0)
    l = np.max(grid_data, axis=0) - origin
    l = np.max(l)
    l = np.linspace(0, l, resolution)

    E = l + origin[0]
    N = l + origin[1]
    grid_interp = np.meshgrid(E, N, indexing='ij')
    interp = np.stack(grid_interp, axis=-1).reshape(-1, 2)

    # interpolate
    d = griddata(grid_data, dstack, interp, method=method)

    if method == 'nearest':
        # remove convex hull (inefficient)
        dLinear = griddata(
            grid_data,
            dstack,
            interp,
            method='linear'
        )
        d[np.isnan(dLinear)] = np.nan

    # reshape to xarray
    d = d.reshape(grid_interp[0].shape)
    d = xr.DataArray(
        d,
        coords={'E': E, 'N': N},
        dims=['E', 'N']
    )

    d['E'] = d['E'] + M[0, 2]
    d['N'] = d['N'] + M[1, 2]
    d = d.transpose('N', 'E')

    return d


def xrfiltfilt(d, f, dim):
    d = xr.apply_ufunc(
        f,
        d,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]])
    return d
