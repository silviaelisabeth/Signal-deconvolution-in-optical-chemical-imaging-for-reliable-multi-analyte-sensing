__author__ = 'Silvia E Zieger'
__project__ = 'multi-analyte imaging using hyperspectral camera systems'

"""Copyright 2020. All rights reserved.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable 
for any damages arising from the use of this software.
Permission is granted to anyone to use this software within the scope of evaluating mutli-analyte sensing. No permission
is granted to use the software for commercial applications, and alter it or redistribute it.

This notice may not be removed or altered from any distribution.
"""

import matplotlib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import matplotlib.patches as patches
from spectral import *
import seaborn as sns
import time
import pathlib
from glob import glob
import h5py
import os.path
import warnings
import xlrd

import layout_plotting_v1_3 as plot

sns.set(style="ticks")
warnings.filterwarnings('ignore')


# =====================================================================================
def _load_reference(file_halogen):
    f = open(file_halogen, "r")
    if 'extrapo' in f.name.split('_')[-1]:
        df_white = pd.read_csv(f, index_col=0, dtype=np.float64)
        df_white.columns = ['Spectral Irrradiance / W/(m²nm)']
    else:
        for en, line in enumerate(f):
            if line.startswith('Wavelength / nm'):
                skiprows_ = en

        df_white = pd.read_csv(file_halogen, skiprows=skiprows_, decimal=',', sep='\t', index_col=0)
    df_white_norm = df_white / df_white.max()
    df_ref = df_white_norm.loc[470:] * 0.95

    return df_ref


def load_testcube(file_hdr, arg, pixel_rot=None, span_area=False):
    if 'show bands' in arg.keys():
        if isinstance(arg['show bands'], float) or isinstance(arg['show bands'], int):
            if int(arg['show bands']) > np.float(open_image(file_hdr).metadata['bands'])-1:
                raise ValueError('Selected wavelength exceeds cube bands of {:.0f}'.format(np.float(open_image(file_hdr).metadata['bands'])-1))
            else:
                img_cube = open_image(file_hdr)[:, :, arg['show bands']]
        elif arg['show bands'] is None:
            img_cube = open_image(file_hdr).open_memmap()
        else:
            if arg['show bands'][1] > np.float(open_image(file_hdr).metadata['bands']):
                raise ValueError('Selected wavelength out of cube range ', open_image(file_hdr).metadata['bands'])
            else:
                img_cube = open_image(file_hdr)[:, :, arg['show bands'][0]:arg['show bands'][1]]

    else:
        img_cube = open_image(file_hdr).open_memmap()

    plotting_cube(img_cube, pixel_rot=pixel_rot, span_area=span_area, arg=arg)


def plotting_cube(img_cube, pixel_rot=None, span_area=False, arg=None):
    # parameter check
    if arg is None:
        px_color = 'k'
        cmap_ = 'jet'
    else:
        if 'px color' in arg.keys():
            px_color = arg['px color']
        else:
            px_color = 'k'

        if 'cmap' in arg.keys():
            cmap_ = str(arg['cmap'])
        else:
            cmap_ = 'jet'

    # load and rotate cube if required
    img_rot = rotation_cube(img_cube=img_cube, arg=arg)
    imshow(img_rot, cmap=cmap_)

    # span area if pixels are given
    if span_area is True:
        if pixel_rot is None:
            raise ValueError('Pixel to span an area are required!')
        if 'facecolor' in arg.keys():
            facecol = arg['facecolor']
        else:
            facecol = 'darkorange'
        if 'alpha area' in arg.keys():
            alpha_ = arg['alpha area']
        else:
            alpha_ = 0.25

        # 4 points per area
        num_area = int(len(pixel_rot)/4)
        for n in range(num_area):
            px = pixel_rot[n*4:(n*4+4)]
            plt.axes().add_patch(patches.Rectangle((px[0][1], px[0][0]),  # (x,y)
                                                   px[2][1] - px[0][1],  # width
                                                   px[2][0] - px[0][0],  # height
                                                   facecolor=facecol, alpha=alpha_))
    else:
        if pixel_rot is None:
            pass
        else:
            for pi in pixel_rot:
                plt.annotate(pi, xy=(pi[1], pi[0]), xycoords='data', xytext=(7, 10), textcoords='offset points',
                             fontsize=8, color=px_color, arrowprops=dict(arrowstyle="->", color=px_color,
                                                                         connectionstyle="arc3"))
    plt.tight_layout()

    return


def _load_cube(file_hdr, correction=True, corr_file=None, rotation_=True, plot_cube=True):
    # load hyperspectral cube
    para_cube = load_cube(file_hdr=file_hdr, corr_file=corr_file, correction=correction, rotation=rotation_,
                          plot_cube=plot_cube)

    # collect data for header
    dic_metadata = dict({'num. bands': para_cube['cube'].metadata['bands'],
                         'TDI steps': para_cube['cube'].metadata['pixel step'],
                         'pixel blur': np.float(para_cube['cube'].metadata['pixel blur']),
                         'Height': para_cube['cube'].metadata['lines'],
                         'Width': para_cube['cube'].metadata['samples'],
                         'Binning WxH': [np.int(para_cube['cube'].metadata['binning columns']),
                                         np.int(para_cube['cube'].metadata['binning rows'])],
                         'all scanned wavelengths': [np.float(i) for i in
                                                     para_cube['cube'].metadata['wavelength']]})

    return para_cube, dic_metadata


# =====================================================================================
def load_correction_v1(corr_file):
    with h5py.File(corr_file, 'r') as f:
        if 'correction factor' in f.keys():
            pass
        else:
            raise ValueError('Correction factor missing in file')

        corr = f['correction factor']
        d = dict(map(lambda k: (np.float(k), corr.get(k).value), corr.keys()))

    return d


def load_cube(file_hdr, rotation=0, correction=True, corr_file=None, plot_cube=True):
    """
    Load the measurement file of the hyperspectral camera.
    :param file_hdr:
    :param rotation:
    :param correction:
    :param corr_file:
    :param plot_cube:
    :return:
    """
    img_cube = open_image(file_hdr)
    integrationtime = np.float(img_cube.metadata['integration time'])
    wavelength = [np.float(l) for l in img_cube.metadata['wavelength']]

    # correction factor
    if correction is True:
        if corr_file is None:
            raise ValueError('Correction file required!')
        df_corr = load_correction_v1(corr_file=corr_file)
    else:
        df_corr = None

    parameter = {'cube': img_cube, 'Integration time': integrationtime, 'Wavelength': wavelength, 'correction': df_corr}

    # plotting cube for verification
    if plot_cube is True:
        img = img_cube.open_memmap()
        if rotation != 0:
            img = img.swapaxes(0, 1)
            img_rot = np.flip(img, 0)
            imshow(img_rot)
        else:
            imshow(img)

    return parameter


def rotation_cube(img_cube, arg):
    if 'rotation' in arg:
        if arg['rotation'] == 0 or arg['rotation'] == 360:
            img = img_cube
        elif arg['rotation'] == 90:
            img_ = img_cube.swapaxes(0, 1)
            img = np.flip(img_, 0)
        elif arg['rotation'] == 180:
            img_ = np.flip(img_cube, 0)
            img = np.flip(img_, 1)
        elif arg['rotation'] == 270:
            img_ = np.flip(img_cube, 0)
            img = img_.swapaxes(0, 1)
    else:
        img = img_cube
    return img


def cube_rearrange(file_hdr):
    # load cube - dictionary with keys cube, Integration time, Wavelength, correction (header and correction factors)
    para = load_cube(file_hdr=file_hdr, correction=False, rotation=False, plot_cube=False)
    itime = str(int(para['Integration time'])) + 'ms'

    # --------------------------------------------------------------
    # preparation for correction
    lambda_meas = [np.float(l) for l in para['cube'].metadata['wavelength']]
    band_meas = para['cube'].open_memmap()

    # dictionary of corrected data; keys = cube-rows; y-axis = cube-samples; x-axis = wavelength
    band_meas = band_meas.swapaxes(0, -1)
    band_meas = band_meas.swapaxes(1, -1)
    dic_cube = dict()
    for en, wl in enumerate(lambda_meas):
        dic_cube[wl] = pd.DataFrame(band_meas[en])

    return para, itime, dic_cube, lambda_meas


def correction_cube(file_hdr, path_corr):
    # load cube - dictionary with keys cube, Integration time, Wavelength, correction (header and correction factors)
    para = load_cube(file_hdr=file_hdr, corr_file=path_corr, rotation=False, plot_cube=False)
    itime = str(int(para['Integration time'])) + 'ms'

    # --------------------------------------------------------------
    # preparation for correction
    lambda_meas = [np.float(l) for l in para['cube'].metadata['wavelength']]

    band_meas = para['cube'].open_memmap()
    # dic_band_meas = dict(map(lambda x: (x, pd.DataFrame(band_meas[x], columns=lambda_meas).T), range(len(band_meas))))

    # dictionary of corrected data; keys = cube-rows; y-axis = cube-samples; x-axis = wavelength
    band_meas = band_meas.swapaxes(0, -1)
    band_meas = band_meas.swapaxes(1, -1)
    wavelength_corr = [i for i in para['correction'].keys()]
    dic_corr = dict()
    for en, wl in enumerate(lambda_meas):
        if wl in wavelength_corr:
            dic_corr[wl] = pd.DataFrame(band_meas[en]).mul(pd.DataFrame(para['correction'][wl]))
    # wavelength_corr = [i for i in para['correction'].keys()] # para['correction']['correction factor'].keys().tolist()
    # dic_corr = dict(map(lambda px: (px[0], list(map(lambda wl:
    #                                                 np.multiply(para['correction']['correction factor'][wl][px[0]],
    #                                                             np.array(px[1].loc[wl])), wavelength_corr))),
    #                     dic_band_meas.items()))

    return para, itime, dic_corr, wavelength_corr


def spectral_deviation(file_sensorID, data_cube, file_ref, name_foile, arg, arg_fit):
    # sensorID stored in xlsx File
    workbook = xlrd.open_workbook(file_sensorID)
    worksheet = workbook.sheet_by_index(0)
    first_row = []  # The row where we stock the name of the column
    for col in range(worksheet.ncols):
        first_row.append(worksheet.cell_value(0, col))
    # transform the workbook to a list of dictionary
    data = []
    for row in range(1, worksheet.nrows):
        elm = {}
        for col in range(worksheet.ncols):
            elm[first_row[col]] = worksheet.cell_value(row, col)
        data.append(elm)
    sensID = pd.concat([pd.DataFrame.from_dict(data[i], orient='index').T for i in range(len(data))])
    sensorID = sensID[['ID', 'indicator dye', 'reference', 'amplification', 'Polymer']].set_index('ID')

    # load reference spectra
    dict_ref = dict()
    for en, f in enumerate(file_ref):
        sens_label = list()
        df_ref_ = pd.read_csv(f, sep=';', skiprows=6, index_col=[0, 1]).T.dropna()
        for s in f.split('/')[-1].split('SZ')[1:]:
            sens_label.append('SZ' + s.split('_')[0])
        df_ref_.columns = sens_label
        ind_int = [np.int(f) for f in df_ref_.index]
        df_ref_.index = ind_int
        dict_ref[name_foile[en]] = df_ref_[name_foile[en]]

    df_ref = pd.DataFrame.from_dict(dict_ref)
    list_ref_label = list()
    for f in name_foile:
        if sensorID.loc[f, 'reference']:
            label_ref = sensorID.loc[f, 'reference'].split('lex')[0][0] + \
                        sensorID.loc[f, 'reference'].split('lex')[1][0]
            label_ref = sensorID.loc[f, 'indicator dye'] + '+' + label_ref
        else:
            label_ref = sensorID.loc[f, 'indicator dye']
        list_ref_label.append(label_ref)
    df_ref.columns = list_ref_label
    df_cube = data_cube.copy()

    # actual plotting
    for m in list_ref_label:
        plot.plot_spectral_deviation(df_ref=df_ref, data_cube=df_cube, dye=m, arg=arg, arg_fit=arg_fit)


def coordinate_rotation(x, y, phi, cube_shape):
    """ coordinate rotation for certain angle (given in deg)
    :param x:
    :param y:
    :param phi: angle for rotation given in deg
    """
    rot_allowed = [0, 90, 180, 270, 360]
    if phi in rot_allowed:
        if phi == 180:
            Y = cube_shape[1] - y
            X = cube_shape[0] - x
        elif phi == 90:
            Y = cube_shape[1] - x
            X = y
        elif phi == 270:
            Y = x
            X = cube_shape[0] - y
        else:  # 0, 360
            X = x * np.cos(np.deg2rad(phi)) + y * np.sin(np.deg2rad(phi))
            Y = -x * np.sin(np.deg2rad(phi)) + y * np.cos(np.deg2rad(phi))
    else:
        raise ValueError("Rotation not possible, please choose one of the following: 0, 90, 180 or 270 deg")
    return np.int(X), np.int(Y)


def split_roi(dic_corr, cube_corr, pixel, name_dyes, averaging=True):
    # dic regions: keys ~ cube-Samples (height in 0deg rotated cube / width in 90deg rotated cube)
    # dataframe for each sensor - region
    if name_dyes is None:
        sensor_tag = ['sensor-' + str(i) for i in np.arange(len(pixel))]
    else:
        sensor_tag = name_dyes

    dic_regions = dict(map(
        lambda en: (sensor_tag[en], dict(map(lambda wl: (wl, dic_corr[wl].loc[pixel[en][0][0]:pixel[en][1][0],
                                                             pixel[en][0][1]:pixel[en][2][1]]),
                                             dic_corr.keys()))), range(len(sensor_tag))))
    cube_corr['pixel of interest'] = pixel
    cube_corr['region of interest'] = dic_regions

    # Averaging
    if averaging is True:
        # creating standard deviation and average value of all RoI
        ind_list = [i.split(' ')[0] for i in name_dyes]
        ind_ls = list(dict.fromkeys(ind_list))

        # mean value between multiple RoI or of one RoI
        dic_mean_ = dict(map(lambda sens: (sens, dict(map(lambda wl: (wl, dic_regions[sens][wl].mean().mean()),
                                                         dic_regions[sens].keys()))),
                            cube_corr['region of interest'].keys()))
        df_av_ = pd.DataFrame(dic_mean_)
        df_mean = pd.concat([df_av_.filter(like=l).mean(axis=1) for l in ind_ls], axis=1)
        df_mean.columns = [s + ' mean' for s in ind_ls]

        # standard deviation within the area
        if len(ind_ls) == len(name_dyes):
            dic_std = dict(map(lambda sens: (sens, dict(map(lambda wl: (wl, dic_regions[sens][wl].std().std()),
                                                            dic_regions[sens].keys()))),
                               cube_corr['region of interest'].keys()))
            df_std = pd.DataFrame(dic_std)
        else:
            df_std = pd.concat([df_av_.filter(like=l).std(axis=1) for l in ind_ls], axis=1)
        df_std.columns = [s + ' STD' for s in ind_ls]
        df_av = pd.concat([df_mean, df_std], axis=1, sort=True)

        cube_corr['average data'] = df_av

    return cube_corr


# =====================================================================================
def correction_factors_hyperCube_v2(file_lamp, file_hdr, save=True):
    """
    for all pixels
    :param file_lamp:
    :param file_hdr:
    :param save:
    :return:
    """
    # load reference spectrum and cube
    df_ref = _load_reference(file_halogen=file_lamp)
    para_cube, dic_metadata = _load_cube(file_hdr=file_hdr, correction=False, rotation_=True, plot_cube=False)

    # extraction of all pixel information and store them into dictionary
    ls_remove = [i for i, e in enumerate(dic_metadata['all scanned wavelengths']) if e == 0]
    l_all = np.arange(0, int(para_cube['cube'].metadata['bands']))
    ls_bands = np.delete(l_all, ls_remove)

    img = open_image(file_hdr)
    dic_band = dict(map(lambda b: (b, img.read_band(b)), ls_bands))

    # ------------------------------------------------------
    # fit reference to cube wavelength
    ls_lambda = dic_metadata['all scanned wavelengths'].copy()
    ls_lambda = list(dict.fromkeys(ls_lambda))
    ls_lambda.remove(0.)
    x_interpol = ls_lambda

    f_interpol = interp1d(df_ref.index, df_ref['Spectral Irrradiance / W/(m²nm)'], kind='cubic')
    y_interpol = f_interpol(x_interpol)

    df_ref_interpol = pd.DataFrame(y_interpol, x_interpol)
    df_ref_interpol.columns = ['Spectral Irrradiance 95% / W/(m²nm)']

    # ------------------------------------------------------
    # correction
    dic_corr = dict(map(lambda b: (dic_metadata['all scanned wavelengths'][b[0]],
                                   df_ref_interpol.loc[dic_metadata['all scanned wavelengths'][b[0]]].values[0] / b[1]),
                        dic_band.items()))

    dic_out = pd.Series({'header': dic_metadata, 'cube': para_cube['cube'], 'correction factor': dic_corr})

    # ----------------------------------------------------------------------------------------------------
    # saving
    if save is True:
        meas_time = file_hdr.split('/')[-1].split('_')[0]
        size = '_cube-' + str(para_cube['cube'].shape[0]) + 'x' + str(para_cube['cube'].shape[1]) + 'px'

        save_name = meas_time + '_correctionfile_TDI' + dic_metadata['TDI steps']
        save_name = save_name + '_blur{:.0f}'.format(dic_metadata['pixel blur']) + size + '.hdf5'

        # --------------------------------------------------
        # create folder is not existent
        file_path = 'correctionFiles/'
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        # ---------------------------------
        # actual saving
        f = h5py.File(file_path + save_name, "w")
        grp_header = f.create_group("header")
        grp_corr = f.create_group("correction factor")

        # save header data
        for k, v in dic_metadata.items():
            grp_header.create_dataset(k, data=v)

        # save correction data
        for k, v in dic_corr.items():
            grp_corr.create_dataset(str(k), data=np.array(v))

        f.close()

    return dic_out


# =====================================================================================
def hyperCube_preparation(file_hdr, arg, unit, name_dyes=None, pixel_rot=None, averaging=False, plotting=False,
                          analyte='O2', save=True, cube_type=None):
    # Cube loading, and split into region of interest
    # --------------------------------------------------------------------------------------------------------------
    # define required parameter
    if (unit in file_hdr) is False:
        conc = np.nan
    else:
        conc = file_hdr.split('_cube')[0].split('_')[-1]

    # ---------------------------------------------------------------------------------
    # correction of the whole cube
    para, itime, dic_cube, wavelength = cube_rearrange(file_hdr=file_hdr)

    # ---------------------------------------------------------------------------------
    # output dictionary
    cube_corr = dict({'Cube': para, 'cube data': dic_cube, 'wavelength': wavelength, 'Concentration': conc})

    # ---------------------------------------------------------------------------------
    # split whole cube into regions of interest
    # coordinate rotation to fit it to the original orientation of the cube
    if pixel_rot:
        if 'rotation' in arg:
            rot = arg['rotation']
        else:
            rot = 0

        pixel_0 = list()
        for px in pixel_rot:
            px_roi = list()
            for p in px:
                px_roi.append(coordinate_rotation(x=p[0], y=p[1], phi=rot, cube_shape=para['cube'].shape))
            if rot == 90:
                px_roi = [px_roi[1], px_roi[-2], px_roi[-1], px_roi[0]]
            if rot == 180:
                px_roi = [px_roi[2], px_roi[-1], px_roi[0], px_roi[1]]
            if rot == 270:
                px_roi = [px_roi[-1], px_roi[0], px_roi[1], px_roi[-2]]
            # rotation for 270deg
            pixel_0.append(px_roi)

        # split cube (original orientation) and corresponding pixel
        cube = split_roi(dic_corr=dic_cube, cube_corr=cube_corr, pixel=pixel_0, name_dyes=name_dyes,
                         averaging=averaging)

    # ---------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        if arg is None:
            figsize_ = (5, 3)
            fontsize_ = 13.
        else:
            figsize_ = arg['figure size meas']
            fontsize_ = arg['fontsize meas']

        plt.ioff()
        fig, ax = plot.plotting_averagedSignal(cube, conc=conc, unit=unit, analyte=analyte, figsize_=figsize_,
                                               fontsize_=fontsize_)
        plt.show()
    else:
        fig = None
        ax = None

    # ---------------------------------------------------------------------------------
    # Saving
    if save is True:
        df_out = pd.Series(cube_corr['region of interest'])
        df_sav = pd.Series({'measurement': file_hdr.split('\\')[-1], 'sensor ID': name_dyes,
                            'concentration': cube['Concentration'], 'region of interest': df_out,
                            'pixel of interest': cube['pixel of interest'], 'wavelength': cube['wavelength']})

        if 'calibration' in file_hdr:
            path_save = file_hdr.split('calibration')[0] + '/output/correctionCube/'
        else:
            path_save = file_hdr.split('measurement')[0] + '/output/correctionCube/measurement/'

        if cube_type == 'single':
            path_save = path_save + 'singleIndicator/'
        elif cube_type == 'multiple':
            path_save = path_save + 'multiIndicator/'
        else:
            raise ValueError('Define whether the cube contains single or multiple indicators')

        if os.path.isdir(path_save) == False:
            pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

        name = file_hdr.split('/')[-1].split('\\')[-1].split('.')[0]
        save_name = path_save + name + '.hdf5'
        if os.path.isfile(save_name) == False:
            pass
        else:
            ls_files_exist = glob(save_name + '*.hdf5')
            f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
            num = 0
            for f in f_exist:
                if 'run' in f:
                    num = int(f.split('run')[-1]) + 1
                else:
                    pass
            save_name = path_save + name + '_run' + str(num) + '.hdf5'

        df_sav.to_hdf(save_name, 'df_sav', format='f')

    return cube, fig, ax


# correction of cube (and) slicing the cube into regions of interest
def correction_hyperCube(file_hdr, path_corr, arg=None, name_dyes=None, pixel_90=None, averaging=False, plotting=False,
                         analyte='O2', unit='%air', save=True):
    """ keys: 'Cube', 'corrected data', 'wavelength', 'Concentration'. If pixel are given, 'pixel of interest', 'region
    of interest' and averaged data if region of interest are selected and averaging is True
    'pixel of interest' are the pixel for the original (not rotated) cube in the shape of (x,y) -
    width (cube-Rows 1300) x height (cube-Samples 1088)
    'region of interest': dictionary for all sensor regions. The keys of the sensor regions correspond to the pixel in
    width-direction which then contain a dataframe with the pixel in height-direction as columns and the wavelength as
    an index
    :param file_hdr:
    :param path_corr:
    :param arg:
    :param name_dyes:
    :param pixel_90:
    :param averaging:
    :param plotting:
    :param save:
    :return:
    """
    # define required parameter
    if (unit in file_hdr) is False:
        conc = np.nan
    else:
        conc = file_hdr.split('_cube')[0].split('_')[-1]

    # ---------------------------------------------------------------------------------
    # correction of the whole cube
    para, itime, dic_corr, wavelength = correction_cube(file_hdr=file_hdr, path_corr=path_corr)

    # ---------------------------------------------------------------------------------
    # output dictionary
    cube_corr = dict({'Cube': para, 'corrected data': dic_corr, 'wavelength': wavelength, 'Concentration': conc})

    # ---------------------------------------------------------------------------------
    # split whole cube into regions of interest
    if pixel_90:
        cube_corr = split_roi(dic_corr=dic_corr, cube_corr=cube_corr, pixel=pixel_90, name_dyes=name_dyes,
                              averaging=averaging)

    # ---------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        if arg is None:
            figsize_ = (5, 3)
            fontsize_ = 13.
        else:
            figsize_ = arg['figure size meas']
            fontsize_ = arg['fontsize meas']

        plt.ioff()
        fig, ax = plot.plotting_averagedSignal(cube_corr, conc=conc, unit=unit, analyte=analyte, figsize_=figsize_,
                                               fontsize_=fontsize_)
        plt.show()
    else:
        fig = None
        ax = None

    # ---------------------------------------------------------------------------------
    # Saving
    if save is True:
        df_out = pd.Series(cube_corr['region of interest'])
        df_sav = pd.Series({'measurement': file_hdr.split('\\')[-1], 'corr file': path_corr, 'sensor ID': name_dyes,
                            'concentration': cube_corr['Concentration'], 'region of interest': df_out,
                            'pixel of interest': cube_corr['pixel of interest'], 'wavelength': cube_corr['wavelength']})

        path_save = file_hdr.split('calibration')[0] + '/output/correctionCube/'
        if os.path.isdir(path_save) == False:
            pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

        name = file_hdr.split('/')[-1].split('\\')[-1].split('.')[0]
        save_name = path_save + name + '.hdf5'
        if os.path.isfile(save_name) == False:
            pass
        else:
            ls_files_exist = glob(save_name + '*.hdf5')
            f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
            num = 0
            for f in f_exist:
                if 'run' in f:
                    num = int(f.split('run')[-1]) + 1
                else:
                    pass
            save_name = path_save + name + '_run' + str(num) + '.hdf5'

        df_sav.to_hdf(save_name, 'df_sav', format='f')

    return cube_corr, fig, ax


def correction_measCube(file_hdr, path_corr, pixel_90, arg=None, name_dyes=None, averaging=False, plotting=False,
                        analyte='O2', unit='%air', save=True):
    """
    keys: 'Cube', 'corrected data', 'wavelength', 'Concentration'. If pixel are given, 'pixel of interest', 'region of
    interest' and averaged data if region of interest are selected and averaging is True
    'pixel of interest' are the pixel for the original (not rotated) cube in the shape of (x,y) -
    width (cube-Rows 1300) x height (cube-Samples 1088)
    'region of interest': dictionary for all sensor regions. The keys of the sensor regions correspond to the pixel in
    width-direction which then contain a dataframe with the pixel in height-direction as columns and the wavelength as
    an index
    :param file_hdr:
    :param path_corr:
    :param arg:
    :param name_dyes:
    :param pixel_90:
    :param averaging:
    :param plotting:
    :return:
    """
    # signal correction - define required parameter
    if (unit in file_hdr) is False:
        conc = np.nan
    else:
        conc = file_hdr.split('_cube')[0].split('_')[-1]

    # ----------------------------------------------------------------------------------------------------------------
    # correction of the whole cube
    para, itime, dic_corr, wavelength = correction_cube(file_hdr=file_hdr, path_corr=path_corr)

    # ----------------------------------------------------------------------------------------------------------------
    # output dictionary
    cube_corr = dict({'Cube': para, 'corrected data': dic_corr, 'wavelength': wavelength, 'Concentration': conc})

    # ----------------------------------------------------------------------------------------------------------------
    # split whole cube into regions of interest
    pixel_0 = []  # shape (x,y): width (cube-Rows 1300) x height (cube-Samples 1088)
    for p in pixel_90:
        px = [(px[1], para['cube'].shape[1] - px[0]) for px in p]
        pixel_0.append([px[1], px[2], px[3], px[0]])

    # dic regions: keys ~ cube-Samples (height in 0deg rotated cube / width in 90deg rotated cube)
    if name_dyes is None:
        sensor_tag = ['sensor-' + str(i) for i in np.arange(len(pixel_90))]
    else:
        sensor_tag = name_dyes

    dic_regions = dict(map(lambda en: (sensor_tag[en],
                                       dict(map(lambda wl: (wl, dic_corr[wl].loc[pixel_0[en][0][0]:pixel_0[en][1][0],
                                                                pixel_0[en][0][1]:pixel_0[en][2][1]]), dic_corr.keys()))),
                           range(len(sensor_tag))))

    cube_corr['pixel of interest'] = pixel_90
    cube_corr['region of interest'] = dic_regions

    # Averaging
    if averaging is True:
        dic_mean = dict.fromkeys(set(cube_corr['region of interest'].keys()))  # dict()
        dic_std = dict.fromkeys(set(cube_corr['region of interest'].keys()))   # dict()
        for sens in cube_corr['region of interest'].keys():
            dic_mean[sens] = dict.fromkeys(set(cube_corr['region of interest'][sens].keys()))   # dict()
            dic_std[sens] = dict.fromkeys(set(cube_corr['region of interest'][sens].keys()))   # dict()
            for px_w in cube_corr['region of interest'][sens].keys():
                dic_mean[sens][px_w] = cube_corr['region of interest'][sens][px_w].mean(axis=1)
                dic_std[sens][px_w] = cube_corr['region of interest'][sens][px_w].std(axis=1)

        df_av = pd.DataFrame(np.zeros(shape=(len(dic_mean[sens][px_w]), len(dic_mean.keys()) * 2)),
                             index=dic_mean[sens][px_w].index)
        ls = []
        for sens in list(dic_mean.keys()):
            ls.append(sens + ' mean')
            ls.append(sens + ' STD')
        df_av.columns = ls

        for s in dic_mean.keys():
            df_av[s + ' mean'] = pd.DataFrame(dic_mean[s]).mean(axis=1)
            df_av[s + ' STD'] = pd.DataFrame(dic_mean[s]).std(axis=1)

        cube_corr['average data'] = df_av

    # ----------------------------------------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        if arg is None:
            figsize_ = (5, 3)
            fontsize_ = 13.
        else:
            figsize_ = arg['figure size meas']
            fontsize_ = arg['fontsize']

        plt.ioff()
        fig, ax = plot.plotting_averagedSignal(cube_corr, conc=conc, unit=unit, analyte=analyte, figsize_=figsize_,
                                               fontsize_=fontsize_)
        plt.show()
    else:
        fig = None
        ax = None

    # ----------------------------------------------------------------------------------------------------------------
    # Saving
    if save is True:
        df_out = pd.Series(cube_corr['region of interest'])
        df_sav = pd.Series({'measurement': file_hdr.split('\\')[-1], 'corr file': path_corr, 'sensor ID': name_dyes,
                            'concentration': cube_corr['Concentration'], 'region of interest': df_out,
                            'pixel of interest': cube_corr['pixel of interest'], 'wavelength': cube_corr['wavelength']})

        # directory
        path_save = file_hdr.split('/')[0] + '/output/measurement/'
        pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

        # file name
        date = str(time.gmtime().tm_year) + str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday) + '_'
        file_name = date + file_hdr.split('/')[-1].split('.')[0]
        save_name = path_save + file_name + '_run0.hdf5'
        if os.path.isfile(save_name) == False:
            pass
        else:
            ls_files_exist = glob(path_save + '*.hdf5')
            f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]

            num = 0
            for f in f_exist:
                if 'run' in f:
                    num = f.split('run')[-1]
                else:
                    pass

            save_name = path_save + file_name + '_run' + str(np.int(num) + 1) + '.hdf5'

        # save to hdf5
        df_sav.to_hdf(save_name, 'df_sav', format='f')

    return cube_corr, fig, ax

