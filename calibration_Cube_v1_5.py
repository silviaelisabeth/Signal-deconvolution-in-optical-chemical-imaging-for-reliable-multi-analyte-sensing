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
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import simps
import numpy as np
import pandas as pd
import random
from lmfit import Model
import time
import scipy.integrate as integrate
from spectral import *
from glob import glob
import pathlib
import os.path
import os
from PIL import Image
import h5py

import curve_fitting_Cube_v1_4 as cC
import correction_hyperCamera_v1_4 as corr
import layout_plotting_v1_3 as plot


# ====================================================================================================================
def load_CorrectedCalibrationFiles(path_calib, arg_fit, unit):
    dic_mean = dict()
    dic_std = dict()
    for i in glob(path_calib + '*.hdf5'):
        t_pc = i.split('_cube')[0].split('_')[-1]
        if t_pc.startswith('0'):
            conc = np.float(t_pc.split(unit)[0]) / 10
        else:
            conc = np.float(t_pc.split(unit)[0])

        if conc > arg_fit['fit concentration'][1]:
            pass
        elif conc < arg_fit['fit concentration'][0]:
            pass
        else:
            df_calib = pd.read_hdf(i, mode='r')
            sensorID = df_calib['sensor ID']
            data = df_calib['region of interest']
            dic_mean[t_pc] = dict(map(lambda s:
                                      (s, pd.DataFrame.from_dict(dict(map(lambda wl: (wl, data[s][wl].mean().mean()),
                                                                          data[s].keys())), orient='index')), sensorID))
            dic_std[t_pc] = dict(map(lambda s:
                                     (s, pd.DataFrame.from_dict(dict(map(lambda wl: (wl, data[s][wl].std().std()),
                                                                         data[s].keys())), orient='index')), sensorID))

    return dic_mean, dic_std


def load_CubeCalibrationFiles(path_calib, name_dyes, pixel_rot, save_cube, arg_fit, unit, analyte='O2'):
    dic_mean = dict()
    dic_std = dict()
    for i in glob(path_calib + '/*hdr'):
        t_pc = i.split('_cube')[0].split('_')[-1]
        if t_pc.startswith('0'):
            conc = np.float(t_pc.split(unit)[0]) / 10
        else:
            conc = np.float(t_pc.split(unit)[0])

        if conc > arg_fit['fit concentration'][1]:
            pass
        elif conc < arg_fit['fit concentration'][0]:
            pass
        else:
            df_calib = corr.hyperCube_preparation(file_hdr=i, arg=None, name_dyes=name_dyes, pixel_rot=pixel_rot,
                                                  unit=unit, analyte=analyte, averaging=True, plotting=False,
                                                  cube_type='multiple', save=save_cube)
            dic_mean[t_pc] = dict(map(lambda s: (s, df_calib[0]['average data'].filter(like='mean')), name_dyes))
            dic_std[t_pc] = dict(map(lambda s: (s, df_calib[0]['average data'].filter(like='STD')), name_dyes))

            for c in dic_mean.keys():
                for s in dic_mean[c].keys():
                    dic_mean[c][s].columns = [0]
                for s in dic_std[c].keys():
                    dic_std[c][s].columns = [0]

    return dic_mean, dic_std


def _load_calibration(path_dual, arg_fit, unit, corrected_cube=False, pixel_rot=None, analyte='O2', path_calib=None,
                      name_dyes=None, save_cube=False):
    if corrected_cube is True:
        # for already corrected cube
        dic_mean, dic_std = load_CorrectedCalibrationFiles(path_calib=path_dual, arg_fit=arg_fit, unit=unit)
    else:
        # for raw cube files when the correction is required
        if all(v is not None for v in [path_calib, name_dyes, pixel_rot]):
            dic_mean, dic_std = load_CubeCalibrationFiles(path_calib=path_calib, arg_fit=arg_fit, name_dyes=name_dyes,
                                                          pixel_rot=pixel_rot, save_cube=save_cube, unit=unit,
                                                          analyte=analyte)
        else:
            raise ValueError('Either provide the directory to the already corrected cube data or provide information '
                             'about the directory to the calibration files, the correction factors and RoI.')
    return dic_mean, dic_std


def _load_calibration_single(path_calib, method, ratiometric, simply):
    ls_calib_files = []
    for i in glob(path_calib + '*.hdf5'):
        if method[:3] in i:
            if ratiometric is True:
                if 'ratiometric' in i.split(method[:3])[1].split('_'):
                    if simply is True:
                        if 'simplifiedSV' in i.split(method[:3])[1].split('_'):
                            ls_calib_files.append(i)
                        else:
                            pass
                    else:
                        if '2sideSV' in i.split(method[:3])[1].split('_'):
                            ls_calib_files.append(i)
                else:
                    pass
            else:
                if 'ratiometric' in i.split(method[:3])[1].split('_'):
                    pass
                else:
                    if simply is True:
                        if 'simplifiedSV' in i.split(method[:3])[1].split('_'):
                            ls_calib_files.append(i)
                        else:
                            pass
                    else:
                        if '2sideSV' in i.split(method[:3])[1].split('_'):
                            ls_calib_files.append(i)

    dic_calib = dict()
    for f in ls_calib_files:
        if 'run' in f.split(method)[-1]:
            dic_calib[f.split('_')[-2]] = pd.read_hdf(f, mode='r')
        else:
            dic_calib[f.split('_')[-1].split('.')[0]] = pd.read_hdf(f, mode='r')
    return dic_calib


def _load_calibration_dual(path_calib, ratiometric, simply):
    ls_calib_files = []
    for i in glob(path_calib + '*.hdf5'):
        if ratiometric is True:
            if 'ratiometric' in i.split('_'):
                if simply is True:
                    if 'simplifiedSV' in i.split('_'):
                        ls_calib_files.append(i)
                    else:
                        pass
                else:
                    if '2sideSV' in i.split('_'):
                        ls_calib_files.append(i)
                    else:
                        pass
            else:
                pass
        else:
            if 'ratiometric' in i.split('_'):
                pass
            else:
                if simply is True:
                    if 'simplified' in i.split('_'):
                        ls_calib_files.append(i)
                    else:
                        pass
                else:
                    if '2sideSV' in i.split('_'):
                        ls_calib_files.append(i)
                    else:
                        pass

    dic_calib = dict()
    for f in ls_calib_files:
        if 'run' in f:
            dic_calib[f.split('_')[-2]] = pd.read_hdf(f, mode='r')
        else:
            dic_calib[f.split('_')[-1].split('.')[0]] = pd.read_hdf(f, mode='r')
    return dic_calib


def _load_generic_function(file_generic, name_single=None):

    df_generic_function = pd.read_csv(file_generic, sep='\t', index_col=0)

    if name_single is None:
        name_single = []
        for s in df_generic_function.columns:
            if 'STD' in s:
                pass
            else:
                name_single.append(s)
    df_generic_corr = df_generic_function.copy()
    df_generic_corr = pd.concat([df_generic_corr.filter(like='mean') - df_generic_corr.filter(like='mean').min(),
                                 df_generic_corr.filter(like='STD')], axis=1, sort=True).sort_index()
    ind_new = [round(i, 4) for i in df_generic_corr.index]
    df_generic_corr.index = ind_new
    # df_generic_corr[name_single] = df_generic_function[name_single] - df_generic_function[name_single].min()

    return df_generic_corr


# -----------------------------------------------------------------
def load_analysis(path_):
    with h5py.File(path_, 'r') as f:
        # load header infos
        header = f['header']
        dic_header = dict(map(lambda k: (k, header.get(k).value), header.keys()))

        # load data
        data = f['data']
        dic_px = dict(map(lambda k: (k, data['pixel'].get(k).value), data['pixel'].keys()))
        sensorID_ = list(dic_header['sensor ID'].split('[')[1].split(']')[0].split(','))
        sensorID = [i.strip()[1:-1] for i in sensorID_]
        col_px = dict(map(lambda m: (m[1].strip(), np.linspace(dic_px['pixel of interest'][m[0]][1][1],
                                                               dic_px['pixel of interest'][m[0]][0][1], endpoint=True,
                                                               dtype=int,
                                                               num=int(dic_px['pixel of interest'][m[0]][0][1] -
                                                                       dic_px['pixel of interest'][m[0]][1][1] + 1))),
                          enumerate(sensorID)))
        index_px = dict(map(lambda m: (m[1].strip(), np.flip(np.linspace(dic_px['pixel of interest'][m[0]][0][0],
                                                                         dic_px['pixel of interest'][m[0]][2][0],
                                                                         endpoint=True, dtype=int,
                                                                         num=int(dic_px['pixel of interest'][m[0]][2][0] -
                                                                                 dic_px['pixel of interest'][m[0]][0][0] + 1)))),
                            enumerate(sensorID)))

        dic_iratio_ = dict(map(lambda k: (k, data['iratio'].get(k).value), data['iratio'].keys()))
        dic_iratio = dict(map(lambda m: (m, pd.concat([pd.DataFrame(dic_iratio_[m][p], index=index_px[m])
                                                       for p in range(len(dic_iratio_[m]))], axis=1, keys=col_px[m])),
                              dic_iratio_.keys()))
        dic_raw_ = dict(map(lambda k: (k, data['rawdata'].get(k).value), data['rawdata'].keys()))
        dic_raw = dict(map(lambda m: (m, pd.concat([pd.DataFrame(dic_raw_[m][p], index=index_px[m])
                                                    for p in range(len(dic_raw_[m]))], axis=1, keys=col_px[m])),
                           dic_raw_.keys()))

        # load results
        results = f['results']
        dic_res_ = dict(map(lambda k: (k, results.get(k).value), results.keys()))
        dic_res = dict()
        for m in dic_res_.keys():
            d = pd.DataFrame([dic_res_[m][p] for p in range(len(dic_res_[m]))])
            d.index = index_px[m]
            d.columns = col_px[m]
            dic_res[m] = d

        return dic_header, col_px, index_px, dic_raw, dic_iratio, dic_res


def load_evaluation(file):
    with h5py.File(file, 'r') as f:
        # load header infos
        header = f['header']
        dic_header = dict(map(lambda k: (k, header.get(k).value), header.keys()))

        # load measurement file name
        file_meas = dic_header['measurement']
        sensID = dic_header['sensor ID'].split("'")[1]
        wl = dic_header['wavelength']
        bnds = dic_header['boundaries linear unmixing']
        calib = dic_header['calibration file']
        fit = dic_header['fitting linear unmixing']
        gen_func = dic_header['generic function']
        ratio = dic_header['ratiometric']

        dict_header = dict({'file raw': file_meas, 'file evaluation': file, 'sensor': sensID, 'wavelength': wl,
                            'boundaries linear unmixing': bnds, 'calibration file': calib, 'fit range': fit,
                            'generic function': fit, 'generic function': gen_func, 'ratiometric evaluation': ratio})

        # --------------------------------------------
        data = f['data']

        # pixel
        dic_px = dict(map(lambda k: (k, data['pixel'].get(k).value), data['pixel'].keys()))
        px = [tuple(dic_px['pixel of interest'][0][o])
              for o in range(len(dic_px['pixel of interest'][0]))]

        # rawdata - (wl, px-H, px-W)
        dic_raw = dict(map(lambda k: (k, data['rawdata'].get(k).value), data['rawdata'].keys()))

        # iratio
        dic_iratio = dict(map(lambda k: (k, data['iratio'].get(k).value), data['iratio'].keys()))
        singleID = list(dic_iratio.keys())

        # calibration data
        dic_calib = dict(map(lambda k: (k, data['calib'].get(k).value), data['calib'].keys()))

        # after 90deg rotation
        pxH_ = np.linspace(start=px[2][1], stop=px[0][1], num=int(np.abs(px[2][1] - px[0][1]) + 1),
                           endpoint=True)
        pxW_ = np.linspace(start=px[2][0], stop=px[0][0], num=int(np.abs(px[2][0] - px[0][0]) + 1),
                           endpoint=True)
        pxH = [int(i) for i in pxH_]
        pxW = [int(i) for i in pxW_]

        iratio1 = pd.DataFrame(dic_iratio[singleID[0]], columns=pxH, index=pxW).sort_index().T
        iratio2 = pd.DataFrame(dic_iratio[singleID[1]], columns=pxH, index=pxW).sort_index().T

        # --------------------------------------------
        res = f['results']
        # RoI label
        dic_ = dict(map(lambda k: (k, res.get(k).value), res.keys()))
        analyte1 = pd.DataFrame(dic_[sensID][0], columns=pxH, index=pxW)
        analyte2 = pd.DataFrame(dic_[sensID][1], columns=pxH, index=pxW)

        # ===========================================================================================
        # load figures
        dic_figures = dict()
        p = file.split("\\")[0] + '/'
        date = file.split("\\")[-1].split('_')[0]
        run = file.split("\\")[-1].split('_run')[1].split('.')[0]

        for p in glob(p + '{}_*run{}*.png'.format(date, run)):
            if singleID[0] in p and 'evaluation' in p:
                im0 = Image.open(p)
                non_transparent0 = Image.new('RGBA', im0.size, (255, 255, 255, 255))
                non_transparent0.paste(im0, (0, 0), im0)
                dic_figures[singleID[0] + '_run{}'.format(run)] = non_transparent0
            if singleID[1] in p and 'evaluation' in p:
                im1 = Image.open(p)
                non_transparent1 = Image.new('RGBA', im1.size, (255, 255, 255, 255))
                non_transparent1.paste(im1, (0, 0), im1)
                dic_figures[singleID[1] + '_run{}'.format(run)] = non_transparent1

        return dict_header, singleID, dic_raw, dic_calib, iratio1, iratio2, analyte1, analyte2, dic_figures


# ====================================================================================================================
def _sternvolmer(x, f, k, m):
    """
    fitting function according to the common two site model. In general, x represents the pO2 or pCO2 content, whereas
    m, k and f are the common fitting parameters
    :param x:   list
    :param m:   np.float
    :param k:   np.float
    :param f:   np.float
    :return:
    """
    # i0/i
    iratio = 1 / (f / (1. + k*x) + (1.-f) / (1. + k*m*x) )
    return iratio


def _sternvolmer_simple(x, f, k):
    """
    fitting function according to the common two site model. In general, x represents the pO2 or pCO2 content, whereas
    m, k and f are the common fitting parameters
    :param x:   list
    :param m:   np.float
    :param k:   np.float
    :param f:   np.float
    :return:
    """
    # i0/i
    iratio = 1 / (f / (1. + k*x) + (1.-f))
    return iratio


def _simplified_SVFit(df_norm, df_sens0, par0=None):

    simply_sv = Model(_sternvolmer_simple)
    if par0:
        params_sens = simply_sv.make_params(k=par0['k'], f=par0['f'])
    else:
        params_sens = simply_sv.make_params(k=0.165, f=0.887)

    params_sens['k'].min = 0.
    params_sens['f'].max = 1.

    # use i0/i data for fit and re-calculate i afterwards
    # full concentration range
    ytofit_sens = 1 / df_norm.values
    xtofit_sens = df_norm.index.to_numpy()
    result = simply_sv.fit(ytofit_sens, params_sens, x=xtofit_sens, nan_policy='omit')

    # 2nd round with one parameter fixed
    params_sens2 = params_sens.copy()
    params_sens2['k'].value = result.best_values['k']
    params_sens2['k'].vary = False
    params_sens2['f'].value = result.best_values['f']
    params_sens2['f'].min = 0.
    params_sens2['f'].max = 1.
    result2 = simply_sv.fit(ytofit_sens, params_sens2, x=xtofit_sens, nan_policy='omit')

    df_bestFit_preSens_norm = pd.DataFrame(1 / result2.best_fit, index=xtofit_sens)
    df_bestFit_preSens = pd.DataFrame(df_sens0 / result2.best_fit, index=xtofit_sens)

    return result2, df_bestFit_preSens, df_bestFit_preSens_norm


def _SVFit(df_norm, df_sens0, par0=None):
    simply_sv = Model(_sternvolmer)
    if par0:
        params_sens = simply_sv.make_params(k=par0['k'], f=par0['f'], m=par0['m'])
    else:
        params_sens = simply_sv.make_params(k=0.165, f=0.887, m=0.05)

    params_sens['k'].min = 0.
    params_sens['f'].max = 1.
    params_sens['m'].min = 0.

    # use i0/i data for fit and re-calculate i afterwards
    # full concentration range
    ytofit_sens = 1 / df_norm.values
    xtofit_sens = df_norm.index.to_numpy()
    result = simply_sv.fit(ytofit_sens, params_sens, x=xtofit_sens, nan_policy='omit')

    # 2nd round with one parameter fixed
    params_sens2 = params_sens.copy()
    params_sens2['k'].value = result.best_values['k']
    params_sens2['k'].vary = False
    params_sens2['f'].value = result.best_values['f']
    params_sens2['f'].min = 0.
    params_sens2['f'].max = 1.
    params_sens2['m'].value = result.best_values['m']
    result2 = simply_sv.fit(ytofit_sens, params_sens2, x=xtofit_sens, nan_policy='omit')

    # 3rd round with another parameter fixed
    params_sens3 = params_sens2.copy()
    params_sens3['k'].value = result2.best_values['k']
    params_sens3['k'].vary = False
    params_sens3['f'].value = result2.best_values['f']
    params_sens3['f'].vary = False
    params_sens3['m'].value = result2.best_values['m']
    result3 = simply_sv.fit(ytofit_sens, params_sens3, x=xtofit_sens, nan_policy='omit')

    df_bestFit_preSens_norm = pd.DataFrame(1 / result3.best_fit, index=xtofit_sens)
    df_bestFit_preSens = pd.DataFrame(df_sens0 / result3.best_fit, index=xtofit_sens)

    return result3, df_bestFit_preSens, df_bestFit_preSens_norm


def SternVolmerFit(dic_integral, df_ref_int, name_dyes, name_single, par0, ref_equal, ratiometric=True, simply=True):
    # pre-check whether enough parameter are defined
    if simply is False:
        for s in name_single:
            if 'm' in par0[s].keys():
                pass
            else:
                print('Provide all starting parameter for 2-side model Stern-Volmer Fit')
                for s in name_single:
                    par0[s]['m'] = 0.05

    dic_out = dict()
    dic_ratio = dict()
    dic_norm = dict()
    for m in name_dyes:
        if ratiometric is True:
            if ref_equal is True:
                df_ratio = dict(map(lambda s: (s, dic_integral[m][s] / df_ref_int.loc[0, m]), name_single))
            else:
                df_ratio = dict(map(lambda s: (s, dic_integral[m][s] / df_ref_int[m]), name_single))
        else:
            df_ratio = dic_integral[m]
        dic_ratio[m] = df_ratio
        df_norm = dict(map(lambda s: (s, df_ratio[s].sort_index() / df_ratio[s].loc[0]), name_single))
        dic_norm[m] = df_norm

        if simply is True:
            out = dict(map(lambda s: (s, _simplified_SVFit(df_norm=df_norm[s], par0=par0[s],
                                                           df_sens0=df_ratio[s].loc[0])), name_single))
        else:
            out = dict(map(lambda s: (s, _SVFit(df_norm=df_norm[s], par0=par0[s], df_sens0=df_ratio[s].loc[0])),
                           name_single))
        dic_out[m] = out

    report = dict(map(lambda m: (m, dict(map(lambda s: (s, dic_out[m][s][0]), name_single))), name_dyes))
    df_bestFit = dict(map(lambda m: (m, dict(map(lambda s: (s, dic_out[m][s][1]), name_single))), name_dyes))
    df_norm = dict(map(lambda m: (m, pd.concat(dict(map(lambda s: (s, dic_out[m][s][2]), name_single)), axis=1)),
                       name_dyes))

    colnew = df_norm[name_dyes[0]].columns.levels[0]
    for m in name_dyes:
        df_norm[m].columns = colnew
    dic_SVFit = dict({'Report': report, 'best Fit': df_bestFit, 'norm best Fit': df_norm, 'data': dic_ratio})

    return dic_SVFit


def SternVolmerFit_v2(df_calib, sensorID, singleID, par0, idxmax_0, arg, ratiometric=True, simply=True):
    # pre-check whether enough parameter are defined
    if simply is False:
        for s in singleID:
            if 'm' in par0[s].keys():
                pass
            else:
                print('Provide all starting parameter for 2-side model Stern-Volmer Fit')
                for s in singleID:
                    par0[s]['m'] = 0.05

    # actual SV Fit
    dic_SVFit = dict()
    dic_norm = dict()
    dic_sens = dict()
    for en, s in enumerate(sensorID):
        if ratiometric is True:
            df_sens = df_calib[s + ' mean'] / df_calib[s + ' ref']
        else:
            df_sens = df_calib[s + ' mean']
        df_norm = df_sens / df_sens.loc[0]
        dic_norm[s] = df_norm
        dic_sens[s] = df_sens

        if simply is True:
            [result, df_bestFitSens, df_bestFitSens_norm] = _simplified_SVFit(df_norm=df_norm, df_sens0=df_sens.loc[0],
                                                                              par0=par0[s.split(' ')[0]])
        else:
            [result, df_bestFitSens, df_bestFitSens_norm] = _SVFit(df_norm=df_norm, df_sens0=df_sens.loc[0],
                                                                   par0=par0[s.split(' ')[0]])

        dic_fit = dict({'Report': result, 'best Fit': df_bestFitSens, 'norm best Fit': df_bestFitSens_norm,
                        'lambda max': idxmax_0[en], 'lambda ref': arg['lambda reference']})
        dic_SVFit[s] = dic_fit

    return dic_SVFit, dic_norm, dic_sens


def SternVolmerFit_integral(dint_sens, dint_ref, sensorID, singleID, par0, ratiometric=True, simply=True):
    if simply is False:
        for s in singleID:
            if 'm' in par0[s].keys():
                pass
            else:
                print('Provide all starting parameter for 2-side model Stern-Volmer Fit')
                for s in singleID:
                    par0[s]['m'] = 0.05

    # actual SV Fit
    dic_SVFit = dict()
    dic_norm = dict()
    dic_sens = dict()
    for s in sensorID:
        if ratiometric is True:
            df_ratio = dint_sens[s] / dint_ref[s]
        else:
            df_ratio = dint_sens[s]
        df_sens = df_ratio.sort_index()
        df_norm = df_sens / df_sens.loc[0]
        dic_norm[s] = df_norm
        dic_sens[s] = df_sens

        if simply is True:
            [result, df_bestFitSens,
             df_bestFitSens_norm] = _simplified_SVFit(df_norm=df_norm[0], df_sens0=df_sens.loc[0, 0],
                                                      par0=par0[s.split(' ')[0]])
        else:
            [result, df_bestFitSens, df_bestFitSens_norm] = _SVFit(df_norm=df_norm[0], df_sens0=df_sens.loc[0, 0],
                                                                   par0=par0[s.split(' ')[0]])

        dic_fit = dict({'Report': result, 'best Fit': df_bestFitSens, 'norm best Fit': df_bestFitSens_norm})
        dic_SVFit[s] = dic_fit

    return dic_SVFit, dic_norm, dic_sens


def conc_SV_simply(k, f, iratio):
    p = 1/k
    denom = 1/iratio - 1 + f
    return (f/denom - 1)*p


def conc_SV(k, f, m, iratio):
    a = k**2 * m
    b = k * ((1 + m) - iratio*(1 + f*(m-1)))
    c = (1 - iratio)

    x1 = (-1*b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    x2 = (-1*b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    # identify physically reasonable values
    if x1.mean().mean() < 0:
        if x2.mean().mean() >= 0:
            x = x2
    elif x2.mean().mean() < 0:
        if x1.mean().mean() >= 0:
            x = x1
    elif x1.mean().mean() < 0 and x2.mean().mean() < 0:
        x = pd.DataFrame(np.nan, index=x1.index, columns=x1.columns)
    else:
        print('Decide for determined oxygen concentration: ', x1, x2)
        x = pd.DataFrame(np.nan, index=x1.index, columns=x1.columns)

    return x


def analyte_concentration_SVFit(dic_calib, dic_ratio, meas_dyes, simply, cutoff=5.):
    # relative intensity ratio i0/i for each measurement point using i0 from SV-Fit
    # Though select the averaged regions for analysis
    bF = dic_calib['bestFit']

    dic_iratio = dict(map(lambda m: (m, bF[m.split(' ')[0] + ' Fit'].loc[0, bF[m.split(' ')[0] + ' Fit'].columns[0]] / dic_ratio[m]),
                          meas_dyes))

    # determination of analyte concentration
    fR = dic_calib['Fitreport']
    if simply is True:
        dic_o2_calc = dict(map(lambda m: (m, conc_SV_simply(k=fR[m.split(' ')[0] + ' Fit'].best_values['k'],
                                                            iratio=dic_iratio[m],
                                                            f=fR[m.split(' ')[0] + ' Fit'].best_values['f'])),
                               meas_dyes))
    else:
        dic_o2_calc = dict(map(lambda m: (m, conc_SV(k=fR[m.split(' ')[0] + ' Fit'].best_values['k'],
                                                     f=fR[m.split(' ')[0] + ' Fit'].best_values['f'],
                                                     m=fR[m.split(' ')[0] + ' Fit'].best_values['m'],
                                                     iratio=dic_iratio[m])), meas_dyes))
    # set negative values as nan
    for m in meas_dyes:
        for col in dic_o2_calc[m]:
            dic_o2_calc[m].loc[~(dic_o2_calc[m][col] > 0), col] = np.nan

    # set values > 10% maximum (calibration) as nan
    O2_max = dic_calib['bestFit'][meas_dyes[0] + ' Fit'].index.max() * (1 + cutoff/100)
    for m in meas_dyes:
        for col in dic_o2_calc[m]:
            dic_o2_calc[m].loc[(dic_o2_calc[m][col] > O2_max), col] = np.nan

    return dic_iratio, dic_o2_calc


def analyte_concentration_SVFit_pixel(dic_iratio, dic_calib, name_dyes, name_single, simply, cutoff=20,
                                      value_check=True):
    # relative intensity ratio i0/i for each pre-factor and pixel using i0 from SV-Fit
    diratio1 = dict(map(lambda m:
                        (m, dic_calib['Fit']['best Fit'][m.split(' ')[0]][name_single[0]].loc[0, 0] / dic_iratio[m].filter(like='a',
                                                                                                                           axis=0)),
                        name_dyes))
    diratio2 = dict(map(lambda m:
                        (m, dic_calib['Fit']['best Fit'][m.split(' ')[0]][name_single[1]].loc[0, 0] / dic_iratio[m].filter(like='b',
                                                                                                                           axis=0)),
                        name_dyes))

    # determine the oxygen concentration in each pixel using one of the calibration curve of the single indicators
    if simply is True:
        # oxygen concentration of single indicator 1
        dic1_o2 = dict(map(lambda m:
                           (m, pd.DataFrame(dict(map(lambda pw:
                                                     (pw[0],
                                                      dict(map(lambda ph:
                                                               (ph, conc_SV_simply(k=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[0]].best_values['k'],
                                                                                   f=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[0]].best_values['f'],
                                                                                   iratio=diratio1[m].loc[pw, ph])),
                                                               diratio1[m].columns))), diratio1[m].index))).T),
                           name_dyes))

        # oxygen concentration of single indicator 2
        dic2_o2 = dict(map(lambda m:
                           (m, pd.DataFrame(dict(map(lambda pw:
                                                     (pw[0],
                                                      dict(map(lambda ph:
                                                               (ph, conc_SV_simply(k=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[1]].best_values['k'],
                                                                                   f=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[1]].best_values['f'],
                                                                                   iratio=diratio2[m].loc[pw, ph])),
                                                               diratio2[m].columns))), diratio2[m].index))).T), name_dyes))
    else:
        # oxygen concentration of single indicator 1
        dic1_o2 = dict(map(lambda m:
                           (m, pd.DataFrame(dict(map(lambda pw:
                                                     (pw[0],
                                                      dict(map(lambda ph:
                                                               (ph, conc_SV(k=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[0]].best_values['k'],
                                                                            f=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[0]].best_values['f'],
                                                                            m=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[0]].best_values['m'],
                                                                            iratio=diratio1[m].loc[pw, ph])),
                                                               diratio1[m].columns))), diratio1[m].index))).T),
                           name_dyes))

        # oxygen concentration of single indicator 2
        dic2_o2 = dict(map(lambda m:
                           (m, pd.DataFrame(dict(map(lambda pw:
                                                     (pw[0],
                                                      dict(map(lambda ph:
                                                               (ph, conc_SV_simply(k=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[1]].best_values['k'],
                                                                                   f=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[1]].best_values['f'],
                                                                                   m=dic_calib['Fit']['Report'][m.split(' ')[0]][name_single[1]].best_values['m'],
                                                                                   iratio=diratio2[m].loc[pw, ph])),
                                                               diratio2[m].columns))), diratio2[m].index))).T),
                           name_dyes))

    # set negative values as nan
    if value_check is True:
        for m in name_dyes:
            for col in dic1_o2[m]:
                dic1_o2[m].loc[~(dic1_o2[m][col] > 0), col] = np.nan
                dic2_o2[m].loc[~(dic2_o2[m][col] > 0), col] = np.nan
    else:
        pass

    # set values > 10% maximum (calibration) as nan
    O2_max = dic_calib['Fit']['data'][name_dyes[0].split(' ')[0]][name_single[0]].index.max() * (1 + cutoff / 100)
    for m in name_dyes:
        for col in dic1_o2[m]:
            dic1_o2[m].loc[~(dic1_o2[m][col] < O2_max), col] = np.nan
            dic2_o2[m].loc[~(dic2_o2[m][col] < O2_max), col] = np.nan

    # store results of individual sensors in joined dictionary
    dic_iratio = dict({name_single[0]: diratio1, name_single[1]: diratio2})
    dic_o2_calc = dict(map(lambda m: (m, dict({name_single[0]: dic1_o2[m], name_single[1]: dic2_o2[m]})), name_dyes))

    return dic_iratio, dic_o2_calc


# =====================================================================================================================
def _crop_data(dic_test, fr):
    arr_test = np.array(list(dic_test.items()))
    arr_test = arr_test.swapaxes(0, -1)
    df_test_re = pd.DataFrame(arr_test).T.set_index(0)
    if fr:
        df_crop = df_test_re.loc[fr[0]:fr[1]]
    else:
        df_crop = df_test_re

    return df_crop


def arrange_data(dt, name_dyes, fr=None):
    dic = dict()
    for m in name_dyes:
        if fr:
            d = _crop_data(dt[m], fr=fr[m])
        else:
            d = _crop_data(dt[m], fr=None)

        wl = d.index.to_numpy()
        key_new = d.loc[wl[0], 1].columns.to_numpy()
        col_new = d.loc[wl[0], 1].index.to_numpy()

        b = [np.array(d.loc[wl, 1]) for wl in d.index]
        B = np.array(b)
        B = B.swapaxes(0, -1)
        B = B.swapaxes(1, -1)

        dic_new = dict(map(lambda px_w: (px_w[1], pd.DataFrame(B[px_w[0]], index=wl, columns=col_new)),
                           enumerate(key_new)))
        dic[m] = dic_new

    return dic


def adjust_outlier_concentration_dic(dic_min, dic_std, ls_conc, name_dyes, unit, ls_outlier=None):
    ls_conc_meas = list([np.int(cm.split(unit)[0]) for cm in dic_min.keys()])

    ls_dup = ls_conc_meas.copy()
    noduples = list(set(ls_conc_meas))
    for x in noduples:
        ls_dup.remove(x)

    dupes = list()
    for x in ls_dup:
        if x in dupes:
            pass
        else:
            dupes.append(x)
    loc_dupes = []
    for x in enumerate(ls_conc_meas):
        if x[1] in dupes:
            loc_dupes.append(x[0])

    # ---------------------------------------------------------------------------------------
    # group and average same concentrations
    if dupes:
        d = dict(map(lambda m: (m, pd.DataFrame(pd.concat([dic_min[list(dic_min.keys())[c]][m] for c in loc_dupes],
                                                      axis=1, sort=True).mean(axis=1))), name_dyes))
        d_std = dict(map(lambda m: (m, pd.DataFrame(pd.concat([dic_std[list(dic_std.keys())[c]][m] for c in loc_dupes],
                                                          axis=1, sort=True).mean(axis=1))), name_dyes))

        # find corresponding reference concentration - closest value to measured concentration
        con_new = min(ls_conc.index.to_numpy() - dupes)

        # update dictionary key - remove duplicates from dictionary and add averaged concentration
        ls_pop = [list(dic_min.keys())[c] for c in loc_dupes]
        [dic_min.pop(i) for i in ls_pop]
        [dic_std.pop(i) for i in ls_pop]

    # remove outlier if requested
    if ls_outlier:
        [dic_min.pop(i) for i in ls_outlier]
        [dic_std.pop(i) for i in ls_outlier]
    else:
        pass

    # rename concentration
    ls_conc_old = list(dic_min.keys())
    ls_conc_old_f = list()
    for c in ls_conc_old:
        ls_conc_old_f.append(np.float(c.split(unit)[0]))
    df_conc_old = pd.DataFrame([ls_conc_old], columns=ls_conc_old_f).sort_index(axis=1, ascending=False)

    ls_conc_ist = pd.DataFrame([sorted(ls_conc.index.to_numpy(), reverse=True)],
                               columns=sorted(noduples, reverse=True))
    for i in df_conc_old.columns:
        dic_min[ls_conc_ist.loc[0, i]] = dic_min.pop(df_conc_old.loc[0, i])
        dic_std[ls_conc_ist.loc[0, i]] = dic_std.pop(df_conc_old.loc[0, i])

    if dupes:
        dic_min[con_new] = d
        dic_std[con_new] = d_std

    return dic_min, dic_std


def averaging(cube_data, name_dyes, arg_fit):
    # averaging data
    dic_data = dict(map(lambda c: (c, dict(map(lambda m:(m, pd.concat([cube_data[c][m][wl].mean(axis=1)
                                                                       for wl in cube_data[c][m].keys()],
                                                                      axis=1).mean(axis=1)), name_dyes))),
                        cube_data.keys()))

    # baseline correction
    fr = arg_fit['fit range sensor']
    dic_min = dict(map(lambda c: (c, dict(map(lambda s: (s, dic_data[c][s] - dic_data[c][s].loc[fr[0]:].min()),
                                              dic_data[c].keys()))), sorted(dic_data.keys())))

    return dic_min


def mean_values_around_nan(dic_index, df):
    for col in dic_index.keys():
        for row in dic_index[col]:
            # average surrounding pixel values
            if col == df.keys()[0]:
                df.loc[row, col] = df.loc[row-1:row+1, col:col+1].mean().mean()
            elif col == df.keys()[-1]:
                df.loc[row, col] = df.loc[row-1:row+1, col-1:col].mean().mean()
            else:
                df.loc[row, col] = df.loc[row-1:row+1, col-1:col+1].mean().mean()
    return df


def _fitting_range(dt, dic_calib, name_dyes):
    fr_s = dict(map(lambda m: (m, (min(dt[m].keys(), key=lambda x: abs(x - dic_calib['info'][m.split(' ')[0] + ' Fit']['sensor'][0])),
                               min(dt[m].keys(), key=lambda x: abs(x - dic_calib['info'][m.split(' ')[0] + ' Fit']['sensor'][1])))),
                name_dyes))
    fr_b = dict(map(lambda m: (m, (min(dt[m].keys(),
                                       key=lambda x: abs(x - dic_calib['info'][m.split(' ')[0] + ' Fit']['background'][0])),
                                   min(dt[m].keys(),
                                       key=lambda x: abs(x - dic_calib['info'][m.split(' ')[0] + ' Fit']['background'][1])))),
                    name_dyes))
    fr_r = dict(map(lambda m: (m, (min(dt[m].keys(),
                                       key=lambda x: abs(x - dic_calib['info'][m.split(' ')[0] + ' Fit']['reference'][0])),
                                   min(dt[m].keys(),
                                       key=lambda x: abs(x - dic_calib['info'][m.split(' ')[0] + ' Fit']['reference'][1])))),
                    name_dyes))
    return fr_s, fr_b, fr_r


# ====================================================================================================================
def save_calibration_singleIndicator(path_calib, dic_SVFit, dic_min, fig, calib_info, name_singles, dic_range, method,
                                     ratiometric, simply, save_op):
    if save_op is None:
        save_op = dict()
        save_op['type'] = ['png']
        save_op['dpi'] = 300

    path_save = path_calib.split('output')[0] + 'output/singleindicator/calibration/'
    if os.path.isdir(path_save) == False:
        pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

    # Saving figures
    if ratiometric is True:
        if simply is True:
            fname = 'Calibration_plot_' + method + '-ratiometric_simplifiedSV'
        else:
            fname = 'Calibration_plot_' + method + '-ratiometric_2sideSV'
    else:
        if simply is True:
            fname = 'Calibration_plot_' + method + '_simplifiedSV'
        else:
            fname = 'Calibration_plot_' + method + '_2sideSV'

    save_name_figure = path_save + fname
    if os.path.isfile(save_name_figure + '.' + save_op['type'][0]) == False:
        pass
    else:
        ls_files_exist = glob(path_save + '*.' + save_op['type'][0])
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
        num = 0
        for f in f_exist:
            if 'run' in f:
                num = int(f.split('run')[-1]) + 1
            else:
                pass
        save_name_figure = path_save + fname + '_run' + str(num)

    for t in save_op['type']:
        fig.savefig(save_name_figure + '.' + t, transparent=False, dpi=save_op['dpi'])

    # preparing data Series for export into hdf5 files
    if method == 'maximal':
        s_out = pd.Series({'sensor wl': calib_info['sensor wl'], 'pre processed data': pd.Series(dic_min),
                           'reference wl': calib_info['reference wl'],
                           'maximal intensity': calib_info['data calibration']})
    elif method == 'integral':
        s_out = pd.Series({'maximal intensity': calib_info, 'pre processed data': pd.Series(dic_min)})

    s_rep = pd.Series()
    s_bFit = pd.Series()
    s_nFit = pd.Series()
    s_div = pd.Series()

    for s in dic_SVFit.keys():
        if s.split(' ')[0] in name_singles:
            s_rep[s] = dic_SVFit[s]['Report']
            s_bFit[s] = dic_SVFit[s]['best Fit']
            s_nFit[s] = dic_SVFit[s]['norm best Fit']

        if method == 'maximal':
            if s.split(' ')[0] in name_singles and s.split(' ')[1].strip() != 'Fit':
                if ratiometric is True:
                    s_div[s] = [dic_SVFit[s]['lambda max'], dic_SVFit[s]['lambda ref']]
                else:
                    s_div[s] = dic_SVFit[s]['lambda max']
        else:
            s_div[s] = dic_range

    if ratiometric is True:
        if simply is True:
            name_data = 'Calibration_' + method + '_ratiometric_simplifiedSV_preprocessed_data'
            name_report = 'Calibration_' + method + '_ratiometric_simplifiedSV_Fitreport'
            name_bestFit = 'Calibration_' + method + '_ratiometric_simplifiedSV_bestFit'
            name_norm = 'Calibration_' + method + '_ratiometric_simplifiedSV_bestFit_normalized'
            name_div = 'Calibration_' + method + '_ratiometric_simplifiedSV_additional_info'
        else:
            name_data = 'Calibration_' + method + '_ratiometric_2sideSV_preprocessed_data'
            name_report = 'Calibration_' + method + '_ratiometric_2sideSV_Fitreport'
            name_bestFit = 'Calibration_' + method + '_ratiometric_2sideSV_bestFit'
            name_norm = 'Calibration_' + method + '_ratiometric_2sideSV_bestFit_normalized'
            name_div = 'Calibration_' + method + '_ratiometric_2sideSV_additional_info'
    else:
        if simply is True:
            name_data = 'Calibration_' + method + '_simplifiedSV_preprocessed_data'
            name_report = 'Calibration_' + method + '_simplifiedSV_Fitreport'
            name_bestFit = 'Calibration_' + method + '_simplifiedSV_bestFit'
            name_norm = 'Calibration_' + method + '_simplifiedSV_bestFit_normalized'
            name_div = 'Calibration_' + method + '_simplifiedSV_additional_info'
        else:
            name_data = 'Calibration_' + method + '_2sideSV_preprocessed_data'
            name_report = 'Calibration_' + method + '_2sideSV_Fitreport'
            name_bestFit = 'Calibration_' + method + '_2sideSV_bestFit'
            name_norm = 'Calibration_' + method + '_2sideSV_bestFit_normalized'
            name_div = 'Calibration_' + method + '_2sideSV_additional_info'

    save_name_data = path_save + name_data + '.hdf5'
    save_name_report = path_save + name_report + '.hdf5'
    save_name_bestFit = path_save + name_bestFit + '.hdf5'
    save_name_norm = path_save + name_norm + '.hdf5'
    save_name_div = path_save + name_div + '.hdf5'

    if os.path.isfile(save_name_data) == False:
        pass
    else:
        ls_files_exist = glob(path_save + '*.hdf5')
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
        num = 0
        for f in f_exist:
            if 'run' in f:
                num = int(f.split('run')[-1]) + 1
            else:
                pass
        save_name_data = path_save + name_data + '_run' + str(num) + '.hdf5'
        save_name_report = path_save + name_report + '_run' + str(num) + '.hdf5'
        save_name_bestFit = path_save + name_bestFit + '_run' + str(num) + '.hdf5'
        save_name_norm = path_save + name_norm + '_run' + str(num) + '.hdf5'
        save_name_div = path_save + name_div + '_run' + str(num) + '.hdf5'

    s_out.to_hdf(save_name_data, 's_out', format='f')
    s_rep.to_hdf(save_name_report, 's_rep', format='f')
    s_bFit.to_hdf(save_name_bestFit, 's_bFit', format='f')
    s_nFit.to_hdf(save_name_norm, 's_nFit', format='f')
    s_div.to_hdf(save_name_div, 's_div', format='f')


def save_calibration_dualIndicator(path_dual, dic_min, param_lc, dic_integral, dic_SVFit, ratiometric, plot_validation,
                                   plotting_fit, standard_dev, simply, fig_val=None, fig_fit=None, save_op=None):
    if save_op is None:
        save_op = dict({'type': ['png', 'jpg'], 'dpi': 300})

    # directory
    if 'calibration' in path_dual:
        path_save = path_dual.split('calibration')[0] + '/output/multiIndicator/calibration/'
    else:
        path_save = path_dual + '/multiIndicator/calibration/'
    if os.path.isdir(path_save) == False:
        pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # saving name figures
    if ratiometric is True:
        if simply is True:
            name_fig_v = 'Dual_indicators_validation_ratiometric_simplifiedSV'
        else:
            name_fig_v = 'Dual_indicators_validation_ratiometric_2sideSV'
    else:
        if simply is True:
            name_fig_v = 'Dual_indicators_validation_simplifiedSV'
        else:
            name_fig_v = 'Dual_indicators_validation_2sideSV'
    save_fig_v = path_save + name_fig_v + '_run0'

    if os.path.isfile(save_fig_v + '.' + save_op['type'][0]) == False:
        pass
    else:
        ls_files_exist = glob(save_fig_v + '.' + save_op['type'][0])
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
        num = 0
        for f in f_exist:
            if 'run' in f:
                num = int(f.split('run')[-1]) + 1
            else:
                pass
        save_fig_v = path_save + name_fig_v + '_run' + str(num)

    if ratiometric is True:
        if simply is True:
            name_fig_c = 'Dual_indicators_calibration_ratiometric_simplifiedSV'
        else:
            name_fig_c = 'Dual_indicators_calibration_ratiometric_2sideSV'
    else:
        if simply is True:
            name_fig_c = 'Dual_indicators_calibration_simplifiedSV'
        else:
            name_fig_c = 'Dual_indicators_calibration_2sideSV'
    save_fig_c = path_save + name_fig_c + '_run0'

    if os.path.isfile(save_fig_c + '.' + save_op['type'][0]) == False:
        pass
    else:
        ls_files_exist = glob(save_fig_c + '.' + save_op['type'][0])
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
        num = 0
        for f in f_exist:
            if 'run' in f:
                num = int(f.split('run')[-1]) + 1
            else:
                pass
        save_fig_c = path_save + name_fig_c + '_run' + str(num)

    for t in save_op['type']:
        if plot_validation is True:
            fig_val.set_size_inches(16, 9)
            fig_val.savefig(save_fig_v + '.' + t, transparent=True, dpi=save_op['dpi'], bbox_inches='tight')
        if plotting_fit is True:
            fig_fit.savefig(save_fig_c + '.' + t, transparent=True, dpi=save_op['dpi'], bbox_inches='tight')

    # ----------------------------------------
    # preparation for export
    s_data = pd.Series({'data': dic_min, 'STD':standard_dev, 'result LC analysis': param_lc, 'integral': dic_integral})
    s_Fit = pd.Series(dic_SVFit)

    # file name
    date = str(time.gmtime().tm_year) + str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday) + '_'

    if ratiometric is True:
        if simply is True:
            file_name_data = date + 'Calibration_dualindicator_ratiometric_simplifiedSV_data'
            file_name_fit = date + 'Calibration_dualindicator_ratiometric_simplifiedSV_Fit'
        else:
            file_name_data = date + 'Calibration_dualindicator_ratiometric_2sideSV_data'
            file_name_fit = date + 'Calibration_dualindicator_ratiometric_2sideSV_Fit'
    else:
        if simply is True:
            file_name_data = date + 'Calibration_dualindicator_simplifiedSV_data'
            file_name_fit = date + 'Calibration_dualindicator_simplifiedSV_Fit'
        else:
            file_name_data = date + 'Calibration_dualindicator_2sideSV_data'
            file_name_fit = date + 'Calibration_dualindicator_2sideSV_Fit'

    save_name_data = path_save + file_name_data + '_run0.hdf5'
    save_name_Fit = path_save + file_name_fit + '_run0.hdf5'

    if os.path.isfile(save_name_data) == False:
        pass
    else:
        if simply is True:
            ls_files_exist = glob(path_save + '*_dualindicator_data*simplified*.hdf5')
        else:
            ls_files_exist = glob(path_save + '*_dualindicator_data*2side*.hdf5')
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]

        num = 0
        for f in f_exist:
            if 'run' in f:
                num = f.split('run')[-1]
            else:
                pass
        save_name_data = path_save + file_name_data + '_run' + str(np.int(num) + 1) + '.hdf5'
        save_name_Fit = path_save + file_name_fit + '_run' + str(np.int(num) + 1) + '.hdf5'

    s_data.to_hdf(save_name_data, 's_data', format='f')
    s_Fit.to_hdf(save_name_Fit, 's_data', format='f')


def save_measurement_dualIndicator(file_meas, path_res, dic_metadata, dic_data, dic_res, ratiometric, simply, dfig_im,
                                   dic_figures, save_op=None):
    if save_op is None:
        save_op = dict({'type': ['png', 'svg'], 'dpi': 300})

    # directory
    if 'output' in path_res:
        path_save = path_res + '/multiIndicator/measurement/'
    else:
        path_save = path_res + '/output/multiIndicator/measurement/'
    if os.path.isdir(path_save) == False:
        pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # saving name data set
    date = str(time.gmtime().tm_year) + str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday) + '_'
    num = 0

    # file name
    if ratiometric is True:
        if simply is True:
            file_name = date + file_meas.split('/')[-1].split('.')[0] + '_simplifiedSV_ratio'
        else:
            file_name = date + file_meas.split('/')[-1].split('.')[0] + '_2sideSV_ratio'
    else:
        if simply is True:
            file_name = date + file_meas.split('/')[-1].split('.')[0] + '_simplifiedSV'
        else:
            file_name = date + file_meas.split('/')[-1].split('.')[0] + '2sideSV'
    save_name = path_save + file_name + '_Analysis_run' + str(num) + '.hdf5'

    if os.path.isfile(save_name) == False:
        pass
    else:
        if simply is True:
            ls_files_exist = glob(path_save + '*simplified*Analysis*.hdf5')
        else:
            ls_files_exist = glob(path_save + '*2side*Analysis*.hdf5')
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]

        for f in f_exist:
            if 'run' in f:
                num = f.split('run')[-1]
            else:
                pass
        save_name = path_save + file_name + '_Analysis_run' + str(np.int(num) + 1) + '.hdf5'

    # actual saving
    f = h5py.File(save_name, "w")
    # group creation
    grp_header = f.create_group("header")
    grp_data = f.create_group("data")
    subgrp_px = grp_data.create_group("pixel")
    subgrp_raw = grp_data.create_group("rawdata")
    subgrp_iratio = grp_data.create_group("iratio")
    subgrp_calib = grp_data.create_group("calib")
    grp_res = f.create_group("results")

    for k, v in dic_metadata.items():
        if k == 'wavelength':
            grp_header.create_dataset(str(k), data=v)
        else:
            grp_header.create_dataset(str(k), data=str(v))

    # save data ROI - raw data, iratio
    subgrp_px.create_dataset('pixel of interest', data=np.array(dic_data['pixel of interest']))
    for k, v in dic_data['raw data RoI'].items():
        ar = np.array(list(map(lambda l: np.array(v[l]), v.keys())))
        subgrp_raw.create_dataset(str(k), data=ar)

    for k, v in dic_data['iratio'].items():
        ar = np.array(list(map(lambda l: np.array(v[l]), v.keys())))
        subgrp_iratio.create_dataset(str(k), data=ar)

    for k, v in dic_data['calibration data'].items():
        if k == 'calib points':
            ar = np.array(v)
            subgrp_calib.create_dataset(str(k), data=ar)
        else:
            ar = np.array(list(map(lambda l: np.array(v[l]), v.keys())))
            subgrp_calib.create_dataset(str(k), data=ar)

    # save results
    for k, v in dic_res.items():
        ar = np.array(list(map(lambda l: np.array(v[l]), v.keys())))
        grp_res.create_dataset(str(k), data=ar)
    f.close()

    # ----------------------------------------
    # saving name figures
    if ratiometric is True:
        if simply is True:
            name_fig_im = date + 'Dualindicator_deconvoluted_ratiometric_simplifiedSV_2Dimage'
            name_fig_op = date + 'Dualindicator_deconvoluted_ratiometric_simplifiedSV_evaluation_2D'
        else:
            name_fig_im = date + 'Dualindicator_deconvoluted_ratiometric_2sideSV_2Dimage'
            name_fig_op = date + 'Dualindicator_deconvoluted_ratiometric_2sideSV_evaluation_2D'
    else:
        if simply is True:
            name_fig_im = date + 'Dualindicator_deconvoluted_simpliefiedSV_2Dimage'
            name_fig_op = date + 'Dualindicator_deconvoluted_simplifiedSV_evaluation_2D'
        else:
            name_fig_im = date + 'Dualindicator_deconvoluted_2sideSV_2Dimage'
            name_fig_op = date + 'Dualindicator_deconvoluted_2sideSV_evaluation_2D'
    save_fig_im = path_save + name_fig_im + '_run' + str(num)
    save_fig_op = path_save + name_fig_op + '_run' + str(num)

    for t in save_op['type']:
        for n in dic_figures.keys():
                dic_figures[n].savefig(save_fig_op + '_' + n + '.' + t, transparent=True, dpi=save_op['dpi'])
        for m in dfig_im.keys():
                dfig_im[m].savefig(save_fig_im + '_' + m + '.' + t, transparent=True, dpi=save_op['dpi'])


def save_measurement_singleIndicator(file_meas, fig_im, fig_op, dic_metadata, dic_data, dic_res, method, ratiometric,
                                     simply, save_op=None):
    if save_op is None:
        save_op = dict({'type': ['png', 'tiff'], 'dpi': 300})

    # directory
    path_save = file_meas.split('measurement')[0] + 'output/singleindicator/measurement/' + method + '/'
    if os.path.isdir(path_save) == False:
        pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # saving name figures
    if ratiometric is True:
        if simply is True:
            name_fig_im = 'Single_indicators_ratiometric_simplifiedSV-evaluation_2D_run'
            name_fig_op = 'Single_indicators_ratiometric_simplifiedSV-evaluation_optodes_run'
        else:
            name_fig_im = 'Single_indicators_ratiometric_2sideSV-evaluation_2D_run'
            name_fig_op = 'Single_indicators_ratiometric_2sideSV-evaluation_optodes_run'
    else:
        if simply is True:
            name_fig_im = 'Single_indicators_simplifiedSV-evaluation_2D_run'
            name_fig_op = 'Single_indicators_simplifiedSV-evaluation_optodes_run'
        else:
            name_fig_im = 'Single_indicators_2sideSV-evaluation_2D_run'
            name_fig_op = 'Single_indicators_2sideSV-evaluation_optodes_run'
    save_fig_im = path_save + name_fig_im

    num = 0
    if os.path.isfile(save_fig_im + str(num) + '.' + save_op['type'][0]) == False:
        pass
    else:
        ls_files_exist = glob(save_fig_im + str(num) + '.' + save_op['type'][0])
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]

        for f in f_exist:
            if 'run' in f:
                num = int(f.split('run')[-1]) + 1
            else:
                pass
    save_fig_im = path_save + name_fig_im + str(num)
    save_fig_op = path_save + name_fig_op + str(num)

    for t in save_op['type']:
        fig_im.savefig(save_fig_im + '.' + t, transparent=True, dpi=save_op['dpi'])
        fig_op.savefig(save_fig_op + '.' + t, transparent=True, dpi=save_op['dpi'])

    # ----------------------------------------
    # saving name data set
    # file name
    date = str(time.gmtime().tm_year) + str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday) + '_'
    file_name = date + file_meas.split('/')[-1].split('.')[0]
    if simply is True:
        save_name = path_save + file_name + '_Analysis_simplifiedSV_run' + str(num) + '.hdf5'
    else:
        save_name = path_save + file_name + '_Analysis_2sideSV_run' + str(num) + '.hdf5'

    if os.path.isfile(save_name) == False:
        pass
    else:
        if simply is True:
            ls_files_exist = glob(path_save + '*Analysis*simplified*.hdf5')
        else:
            ls_files_exist = glob(path_save + '*Analysis*2side*.hdf5')
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]

        num = 0
        for f in f_exist:
            if 'run' in f:
                num = f.split('run')[-1]
            else:
                pass
        save_name = path_save + file_name + '_run' + str(np.int(num) + 1) + '.hdf5'

    # actual saving
    f = h5py.File(save_name, "w")
    # group creation
    grp_header = f.create_group("header")
    grp_data = f.create_group("data")
    subgrp_px = grp_data.create_group("pixel")
    subgrp_raw = grp_data.create_group("rawdata")
    subgrp_iratio = grp_data.create_group("iratio")
    grp_res = f.create_group("results")

    for k, v in dic_metadata.items():
        if k == 'wavelength':
            grp_header.create_dataset(str(k), data=v)
        else:
            grp_header.create_dataset(str(k), data=str(v))

    # save data ROI - raw data, iratio
    subgrp_px.create_dataset('pixel of interest', data=np.array(dic_data['pixel of interest']))
    for k, v in dic_data['raw data RoI'].items():
        ar = np.array(list(map(lambda l: np.array(v[l]), v.keys())))
        subgrp_raw.create_dataset(str(k), data=ar)
    for k, v in dic_data['iratio'].items():
        ar = np.array(list(map(lambda l: np.array(v[l]), v.keys())))
        subgrp_iratio.create_dataset(str(k), data=ar)

    # save results
    for k, v in dic_res.items():
        ar = np.array(list(map(lambda l: np.array(v[l]), v.keys())))
        grp_res.create_dataset(str(k), data=ar)
    f.close()


# ====================================================================================================================
def singleSensor_calibration_maximal(dic_min, dic_calib_STD, sensorID, singleID, arg, arg_fit, analyte, unit, par0=None,
                                     plotting=True, ratiometric=True, simply=True):
    """
    Use of already corrected Cubes.
    :param dic_min:
    :param dic_calib_STD:
    :param sensorID:
    :param arg:
    :param ls_conc:
    :param plotting:
    :return:
    """
    col_header = []
    for l in sensorID:
        col_header.append(l + ' mean')
        col_header.append(l + ' STD')
    df_calib = pd.DataFrame(np.zeros(shape=(len(dic_min.keys()), len(col_header))), index=sorted(dic_min.keys()),
                            columns=col_header)

    idxmax_0 = [dic_min[0][i].loc[arg_fit['fit range sensor'][0]:arg_fit['fit range sensor'][1]].idxmax()
                for i in sensorID]

    idx_ref = list()
    for en, s in enumerate(sensorID):
        for em, c in enumerate(sorted(dic_min.keys())):
            if ratiometric is True:
                wl_ref = min(dic_min[c][s].index.tolist(), key=lambda x: abs(x - arg['lambda reference']))

                df_calib.loc[c, s + ' ref'] = dic_min[c][s].loc[wl_ref].values[0]

            df_calib.loc[c, s + ' mean'] = dic_min[c][s].loc[idxmax_0[en]].values[0]
            df_calib.loc[c, s + ' STD'] = dic_calib_STD[c][s].loc[idxmax_0[en]].values[0]
        if ratiometric is True:
            idx_ref.append(wl_ref)

    # --------------------------------------------------------------------
    # Stern-Volmer Fit
    [dic_SVFit, dic_norm,
     dic_sens] = SternVolmerFit_v2(df_calib=df_calib, sensorID=sensorID, singleID=singleID, par0=par0, simply=simply,
                                   arg=arg, idxmax_0=idxmax_0, ratiometric=ratiometric)

    # --------------------------------------------------------------------
    # combine information from RoI (average and STD) for each single indicator
    col_ls = [i + ' mean' for i in singleID] + [i + ' STD' for i in singleID]
    df_norm_av = pd.DataFrame(np.zeros(shape=(0, len(singleID)*2)), columns=col_ls)
    df_data = df_norm_av.copy()

    for s in singleID:
        l = pd.concat([dic_norm[st] for st in dic_norm.keys() if st.split(' ')[0] == s], axis=1)
        df = pd.concat([dic_sens[st] for st in dic_sens.keys() if st.split(' ')[0] == s], axis=1)
        df_norm_av[s + ' mean'] = l.mean(axis=1)
        df_norm_av[s + ' STD'] = l.std(axis=1)
        df_data[s + ' mean'] = df.mean(axis=1)
        df_data[s + ' STD'] = df.std(axis=1)

        if simply is True:
            [res_av, bestFitSens_av,
             bestFitSens_av_norm] = _simplified_SVFit(df_norm=df_norm_av[s + ' mean'], par0=par0[s],
                                                      df_sens0=df_data[s + ' mean'].loc[0])
        else:
            if 'm' in par0[s].keys():
                pass
            else:
                print('Provide all starting parameter for 2-side model Stern-Volmer Fit')
                for s in singleID:
                    par0[s]['m'] = 0.05
            [res_av, bestFitSens_av,
             bestFitSens_av_norm] = _SVFit(df_norm=df_norm_av[s + ' mean'], df_sens0=df_data[s + ' mean'].loc[0],
                                           par0=par0[s])

        dic_SVFit['data norm averaged'] = df_norm_av
        dic_SVFit['data averaged'] = df_data
        dic_av = {'Report': res_av, 'best Fit': bestFitSens_av, 'norm best Fit': bestFitSens_av_norm}
        dic_SVFit[s + ' Fit'] = dic_av

    # --------------------------------------------------------------------
    plt.ioff()
    fig, ax = plot.plotting_singleFit_maximal(dic_SVFit=dic_SVFit, df_calib=df_calib, singleID=singleID, arg=arg,
                                              simply=simply, unit=unit, analyte=analyte, ratiometric=ratiometric)
    if plotting is True:
        plt.show()
    else:
        plt.close(fig)

    # --------------------------------------------------------------------
    # combine information for calibration
    dic_calib = dict({'sensor wl': [i[0] for i in idxmax_0], 'reference wl': idx_ref, 'data calibration': df_calib})

    return dic_SVFit, dic_calib, fig, ax


# integral without curve fitting -> simpson's rule for sample data
def singleSensor_calibration_integral(sensorID, singleID, dic_min, par0, what_fit, arg, arg_fit, analyte, unit,
                                      ratiometric=True, plotting=False, simply=True):
    # define wavelength range for fitting
    fr_s = arg_fit['fit range sensor']
    if 'reference' in what_fit:
        fr_r = arg_fit['fit range ref']
        if fr_s[0] < fr_r[0]:
            ind_rest = (fr_s[1], fr_r[0])
        else:
            ind_rest = (fr_r[1], fr_s[0])
    else:
        fr_r = None
        ind_rest = (fr_s[0] - 150, fr_s[0])
    dic_range = dict({'sensor': fr_s, 'background': ind_rest, 'reference': fr_r})

    # select wavelength range for sensor, background and (if required) reference
    dic_sens = dict(map(lambda s: (s, pd.concat(dict(map(lambda c: (c, dic_min[c][s].loc[fr_s[0]:fr_s[1]]),
                                                         dic_min.keys())), axis=1)), sensorID))

    if 'reference' in what_fit:
        dic_ref = dict(map(lambda s: (s, pd.concat(dict(map(lambda c: (c, dic_min[c][s].loc[fr_r[0]:fr_r[1]]),
                                                            dic_min.keys())), axis=1)), sensorID))

    # ----------------------------------------------------------------------
    #  Integral determination using simpson's rule for data samples
    dint_sens = dict(map(lambda m: (m, pd.DataFrame(list(map(lambda c:
                                                             (simps(dic_sens[m][c][0].to_numpy(),
                                                                    dic_sens[m].index.to_numpy())),
                                                             dic_sens[m].columns.levels[0])),
                                                    index=dic_sens[m].columns.levels[0])), sensorID))
    dint_ref = dict(map(lambda m: (m, pd.DataFrame(list(map(lambda c:
                                                            (simps(dic_ref[m][c][0].to_numpy(),
                                                                   dic_ref[m].index.to_numpy())),
                                                            dic_ref[m].columns.levels[0])),
                                                   index=dic_ref[m].columns.levels[0])), sensorID))
    dintegral = dict({'sensor': dint_sens, 'reference': dint_ref})

    # --------------------------------------------------------------------
    # Stern-Volmer Fit
    [dic_SVFit, dic_norm,
     dic_sens] = SternVolmerFit_integral(dint_sens=dint_sens, dint_ref=dint_ref, sensorID=sensorID, singleID=singleID,
                                         par0=par0, ratiometric=ratiometric, simply=simply)

    # --------------------------------------------------------------------
    # combine information from RoI (average and STD) for each single indicator
    col_ls = [i + ' mean' for i in singleID] + [i + ' STD' for i in singleID]
    df_norm_av = pd.DataFrame(np.zeros(shape=(0, len(singleID)*2)), columns=col_ls)
    df_data = df_norm_av.copy()
    for s in singleID:
        l = pd.concat([dic_norm[st] for st in dic_norm.keys() if st.split(' ')[0] == s], axis=1)
        df = pd.concat([dic_sens[st] for st in dic_sens.keys() if st.split(' ')[0] == s], axis=1)
        df_norm_av[s + ' mean'] = l.mean(axis=1)
        df_norm_av[s + ' STD'] = l.std(axis=1)
        df_data[s + ' mean'] = df.mean(axis=1)
        df_data[s + ' STD'] = df.std(axis=1)

        if simply is True:
            [res_av, bestFitSens_av,
             bestFitSens_av_norm] = _simplified_SVFit(df_norm=df_norm_av[s + ' mean'], par0=par0[s],
                                                      df_sens0=df_data[s + ' mean'].loc[0])
        else:
            if 'm' in par0[s].keys():
                pass
            else:
                print('Provide all starting parameter for 2-side model Stern-Volmer Fit')
                for s in singleID:
                    par0[s]['m'] = 0.05
            [res_av, bestFitSens_av,
             bestFitSens_av_norm] = _SVFit(df_norm=df_norm_av[s + ' mean'], df_sens0=df_data[s + ' mean'].loc[0],
                                           par0=par0[s])

        dic_SVFit['data norm averaged'] = df_norm_av
        dic_SVFit['data averaged'] = df_data
        dic_av = {'Report': res_av, 'best Fit': bestFitSens_av, 'norm best Fit': bestFitSens_av_norm}
        dic_SVFit[s + ' Fit'] = dic_av

    # --------------------------------------------------------------------
    # Plotting
    plt.ioff()
    fig, ax = plot.plotting_singleFit_integral(dic_SVFit=dic_SVFit, df_data=df_data, df_norm_av=df_norm_av, arg=arg,
                                               singleID=singleID, simply=simply, unit=unit, analyte=analyte,
                                               ratiometric=ratiometric)
    if plotting is True:
        plt.show()
    else:
        plt.close(fig)

    return dic_SVFit, dintegral, fig, ax, dic_range


def singleSensor_measurement_maximal(cube, dic_calib, meas_dyes, ratiometric):
    RoI = cube['region of interest']

    # extract maximal intensity-wavelength from calibration - sensor and reference region
    wl_s = dict(map(lambda m: (m[1].split(' ')[0], dic_calib['data']['sensor wl'][m[0]]),
                    enumerate(dic_calib['info'].keys())))
    wl_r = dict(map(lambda m: (m[1].split(' ')[0], dic_calib['data']['reference wl'][m[0]]),
                    enumerate(dic_calib['info'].keys())))

    # crop measurement region
    dic_meas = dict(map(lambda r: (r[1], RoI[r[1]][wl_s[r[1].split(' ')[0]]]), enumerate(RoI.keys())))

    # in case crop region of reference dye
    if ratiometric is True:
        dic_ref = dict(map(lambda r: (r[1], RoI[r[1]][wl_r[r[1].split(' ')[0]]]), enumerate(RoI.keys())))
        dic_ratio = dict(map(lambda m: (m, dic_meas[m] / dic_ref[m]), meas_dyes))
    else:
        dic_ref = None
        dic_ratio = dict(map(lambda m: (m, dic_meas[m])))

    return dic_ratio, dic_meas, dic_ref


def singleSensor_measurement_integral(cube, dic_calib, name_dyes, ratiometric):
    # slice cube to required wavelength range
    RoI = cube['region of interest']

    # find closest value in list - define fitting range for integral
    fr_s, fr_b, fr_r = _fitting_range(dt=RoI, dic_calib=dic_calib, name_dyes=name_dyes)

    # actual cropping of data arrangement according to defined wavelength range
    dic_sens = arrange_data(dt=RoI, fr=fr_s, name_dyes=name_dyes)
    if ratiometric is True:
        dic_ref = arrange_data(dt=RoI, fr=fr_r, name_dyes=name_dyes)
    else:
        dic_ref = None

    # ----------------------------------------------------------------------
    # Integral using Simpson's rule for data samples
    dint_sens = dict(map(lambda m:
                         (m, pd.concat(dict(map(lambda pw: (pw, pd.DataFrame([simps(dic_sens[m][pw][ph].to_numpy(),
                                                                                    dic_sens[m][pw][ph].index.to_numpy())
                                                                              for ph in dic_sens[m][pw].columns],
                                                                             index=dic_sens[m][pw].columns)),
                                                dic_sens[m].keys())), axis=1)), name_dyes))

    if ratiometric is True:
        dint_ref = dict(map(lambda m:
                            (m, pd.concat(dict(map(lambda pw: (pw, pd.DataFrame([simps(dic_ref[m][pw][ph].to_numpy(),
                                                                                       dic_ref[m][pw][
                                                                                           ph].index.to_numpy())
                                                                                 for ph in dic_ref[m][pw].columns],
                                                                                index=dic_ref[m][pw].columns)),
                                                   dic_ref[m].keys())), axis=1)), name_dyes))

        dic_ratio = dict(map(lambda m: (m, dint_sens[m] / dint_ref[m]), name_dyes))
    else:
        dic_ratio = dint_sens
        dint_ref = None

    for m in name_dyes:
        dint_sens[m].columns = dint_sens[m].columns.levels[0]
        dic_ratio[m].columns = dic_ratio[m].columns.levels[0]
    if ratiometric is True:
        dint_ref[m].columns = dint_ref[m].columns.levels[0]

    return dic_ratio, dint_sens, dint_ref


# -----------------------------------------------------------------
def singleSensor_calibration(path_calib, arg, arg_fit, name_RoI, name_singles, unit, analyte, save_op=None,
                             path_firesting=None, plotting=True, save=False, what_fit=None, par0=None, ls_outlier=None,
                             ratiometric=True, method='maximal', simply=True):
    # load calibration files
    dic_calib, dic_calib_STD = load_CorrectedCalibrationFiles(path_calib=path_calib, arg_fit=arg_fit, unit=unit)
    sensorID = [i for i in dic_calib[list(dic_calib.keys())[0]].keys()]

    # ---------------------------------------------------------------------------
    # concentration of the analyte measured by the reference sensor
    if path_firesting is None:
        ls_conc = dic_calib.keys().tolist()
    else:
        ls_conc = pd.read_csv(path_firesting, sep='\t', usecols=[1, 2], index_col=0).sort_index()

    ls_conc = ls_conc.loc[arg_fit['fit concentration'][0]: arg_fit['fit concentration'][1]*1.05]

    # ---------------------------------------------------------------------------
    # remove outlier and prepare dictionary concentration (in case average replications of concentration points)
    dic_min, dic_calib_STD = adjust_outlier_concentration_dic(dic_min=dic_calib, dic_std=dic_calib_STD, ls_conc=ls_conc,
                                                              name_dyes=name_RoI, ls_outlier=ls_outlier, unit=unit)

    # ---------------------------------------------------------------------------
    # Evaluation either using maximal intensity or integral
    if method == 'maximal':
        # maximal intensity needs only measurement data not fitted data
        [dic_SVFit, calib_info, fig,
         ax] = singleSensor_calibration_maximal(dic_min=dic_min, dic_calib_STD=dic_calib_STD, sensorID=sensorID,
                                                arg=arg, plotting=plotting, par0=par0, ratiometric=ratiometric,
                                                singleID=name_singles, arg_fit=arg_fit, analyte=analyte, unit=unit,
                                                simply=simply)
        dic_range = None
    elif method == 'integral':
        # curve fitting required - v2 contains a sample integration according to Simpson's rule for data samples
        if what_fit is None:
            raise ValueError('Provide information about what to fit! Sensor + background with or without reference?')
        [dic_SVFit, calib_info, fig, ax,
         dic_range] = singleSensor_calibration_integral(sensorID=sensorID, dic_min=dic_min, par0=par0, unit=unit,
                                                        arg=arg, what_fit=what_fit, arg_fit=arg_fit, plotting=plotting,
                                                        ratiometric=ratiometric, singleID=name_singles, analyte=analyte,
                                                        simply=simply)
    else:
        raise ValueError('Decide on the evaluation method! Either use maximal or integral.')

    # --------------------------------------------------------------------
    # saving results
    print(method)
    if save is True:
        save_calibration_singleIndicator(path_calib=path_calib, dic_SVFit=dic_SVFit, dic_min=dic_min, fig=fig,
                                         calib_info=calib_info, name_singles=name_singles, dic_range=dic_range,
                                         method=method, ratiometric=ratiometric, simply=simply, save_op=save_op)

    return dic_SVFit, calib_info, dic_calib, dic_min


def measurement_evaluation_single(file_meas, path_calib, meas_dyes, name_ind, pixel_rot, arg_meas, unit='%air',
                                  analyte='O2', method='maximal', ratiometric=True, plotting=True, saving=False,
                                  save_op=None, save_RoI=True, cmap='inferno', cutoff=5, simply=True):
    # correction of cube and select regions of interest
    cube, fig, ax = corr.hyperCube_preparation(file_hdr=file_meas, arg=arg_meas, name_dyes=meas_dyes, unit=unit,
                                               pixel_rot=pixel_rot, averaging=True, plotting=False, save=save_RoI,
                                               analyte=analyte, cube_type='single')

    # -----------------------------------------------------------------------------------------------------
    # load calibration and prepare calibration plot
    dic_calib = _load_calibration_single(path_calib=path_calib, method=method, ratiometric=ratiometric, simply=simply)
    ind_calib = [l.split(' ')[0] for l in dic_calib['normalized'].keys()]
    ind_calib = list(dict.fromkeys(ind_calib))

    # -----------------------------------------------------------------------------------------------
    # check whether the sensor is the same as the calibration file
    for m in meas_dyes:
        if m.split(' ')[0] in ind_calib:
            pass
        else:
            raise ValueError('Could not find the sensor {} in the calibration file. Only '.format(m),
                             dic_calib['bestFit'].keys(), ' are available')

    # -----------------------------------------------------------------------------------------------
    # extraction of measurement and reference information (maximal intensity or integral) and signal pre-processing
    if method == 'maximal':
        [dic_ratio, dic_sens,
         dic_ref] = singleSensor_measurement_maximal(cube=cube, dic_calib=dic_calib, meas_dyes=meas_dyes,
                                                     ratiometric=ratiometric)
    else:
        [dic_ratio, dic_sens,
         dic_ref] = singleSensor_measurement_integral(cube=cube, dic_calib=dic_calib, name_dyes=meas_dyes,
                                                      ratiometric=ratiometric)

    # -----------------------------------------------------------------------------------------------
    # Stern-Volmer Fit for analyte determination
    dic_iratio, dic_o2_calc = analyte_concentration_SVFit(dic_calib=dic_calib, dic_ratio=dic_ratio, meas_dyes=meas_dyes,
                                                          cutoff=cutoff, simply=simply)

    # -----------------------------------------------------------------------------------------------
    # find position (x, y) of nan and use average of surrounding pixel values for all single indicators
    dic_calc = dict()
    for s in meas_dyes:
        df = dic_o2_calc[s].copy()
        # check whether the dataframe has nan values
        if df.isnull().any().any() == True:
            # find position of nan values
            dic_index = dict(map(lambda c: (c, df[df[c].isnull()].index.tolist()), df.columns))

            # go through each row of each column
            df = mean_values_around_nan(dic_index, df)
        dic_calc[s] = df

    # -----------------------------------------------------------------------------------------------
    # re-combine RoI for each single indicator
    dict_comb = dic_calc.copy()
    for en, n in enumerate(name_ind):
        list_comb = list()
        for l in dic_calc.keys():
            if l.split(' ')[0] == n:
                list_comb.append(l)
        dict_comb[n] = list_comb
    dict_combined = dict(map(lambda i: (i, pd.concat([dic_calc[r] for r in dict_comb[i]], axis=0).sort_index()),
                             name_ind))

    # -----------------------------------------------------------------------------------------------------
    # Plotting
    img = open_image(file_meas)[:, :, np.int(cube['Cube']['cube'].metadata['bands']) - 10]
    px_y = np.arange(0, img.shape[0])
    px_x = np.arange(0, img.shape[1])

    cube_plot = img.reshape(img.shape[0], img.shape[1])
    Xcube, Ycube = np.meshgrid(px_x, px_y)

    # create image frame with all values zero besides the one from the RoI
    frame = pd.DataFrame(np.zeros(shape=cube['Cube']['cube'].shape[:-1]))  # original orientation of cube
    for s in name_ind:
        frame.loc[dict_combined[s].index[0]:dict_combined[s].index[-1],
        dict_combined[s].columns[0]:dict_combined[s].columns[-1]] = dict_combined[s]

    # Plotting the image overlaid with results
    plt.ioff()
    fig_im = plot.plotting_image_overlay(name_dyes=meas_dyes, Xcube=Xcube, Ycube=Ycube, cube_plot=cube_plot,
                                         dic_calc=dic_calc, sensortype='single', cmap=cmap, frame=frame,
                                         dic_calib=dic_calib)

    # Plotting only the optode(s)
    plt.ioff()
    fig_op = plot.plotting_optodes(indicators_calib=ind_calib, meas_dyes=meas_dyes, cmap=cmap, dic_calc=dic_o2_calc,
                                   sensortype='single', indicator=None, frame=frame, dic_calib=dic_calib, unit=unit,
                                   analyte=analyte)
    if plotting is True:
        plt.show()
    else:
        plt.close(fig_im)
        plt.close(fig_op)

    # ------------------------------------------------------------------
    # saving
    if saving is True:
        # preparation of data set for export
        # header
        dic_metadata = dict({'measurement': file_meas.split('\\')[-1], 'sensor ID': meas_dyes,
                             'concentration': cube['Concentration'], 'wavelength': cube['wavelength']})

        # data
        dic_data = dict({'pixel of interest': cube['pixel of interest'], 'raw data RoI': cube['region of interest'],
                         'iratio': dic_iratio})

        # results
        dic_res = dict(map(lambda m: (m, dic_calc[m]), meas_dyes))

        # actual saving
        save_measurement_singleIndicator(file_meas=file_meas, fig_im=fig_im, fig_op=fig_op, dic_metadata=dic_metadata,
                                         dic_data=dic_data, dic_res=dic_res, save_op=save_op, method=method,
                                         ratiometric=ratiometric, simply=simply)

    return cube, dic_iratio, dic_calc, dict_combined


# ====================================================================================================================
def integral_reference(dic_data, arg_fit, name_dyes, ref_equal=False):
    lc = arg_fit['fit range ref']
    dic_ref = dict(map(lambda s: (s, pd.concat(dict(map(lambda c: (c, dic_data[c][s].loc[lc[0]:lc[1]]),
                                                        dic_data.keys())), axis=1)), name_dyes))

    # assuming the reference remain constant over the concentration variation
    if ref_equal is True:
        dint = dict(map(lambda m: (m, simps(dic_ref[m].mean(axis=1).to_numpy(), dic_ref[m].index.to_numpy())),
                        name_dyes))

        df_ref_int = pd.DataFrame.from_dict(dint, orient='index').T
    else:
        # determine integral for each concentration separate
        dint = dict(map(lambda m: (m, dict(map(lambda c: (c, simps(dic_ref[m][c].to_numpy(),
                                                                   dic_ref[m].index.to_numpy())), dic_ref[m].columns))),
                        name_dyes))

        df_ref_int = pd.DataFrame(np.zeros(shape=(len(dic_ref[name_dyes[0]].columns), 0)),
                                  index=dic_ref[name_dyes[0]].columns)
        for m in name_dyes:
            d = pd.DataFrame.from_dict(dint[m], orient='index')
            d.columns = [m]
            df_ref_int[m] = d

    return df_ref_int


def integral_reference_pixel(dic_data, arg_fit, name_dyes, ref_equal=False):
    lc = arg_fit['fit range ref']
    dic_ref = dict(map(lambda m: (m, dict(map(lambda pw: (pw, dic_data[m][pw].loc[lc[0]:lc[1]]),
                                              list(dic_data[m].keys())))), name_dyes))

    # actual integral of the reference region using Simpson's rule (scipy)
    if ref_equal is True:
        # assuming the reference remain constant over the concentration variation
        dint = dict(map(lambda m:
                        (m, pd.DataFrame.from_dict(dict(map(lambda pw:
                                                            (pw, simps(dic_ref[m][pw].mean(axis=1).to_numpy(),
                                                                       dic_ref[m][pw].index.to_numpy())),
                                                            list(dic_data[m].keys()))), orient='index').mean()),
                        name_dyes))
        df_ref_int = pd.DataFrame.from_dict(dint).T
    else:
        # determine integral for each pixel separately
        dint = dict(map(lambda m:
                        (m, pd.concat(dict(map(lambda pw:
                                               (pw, pd.DataFrame.from_dict(dict(map(lambda ph:
                                                                                    (ph, simps(dic_ref[m][pw][ph].to_numpy(),
                                                                                               dic_ref[m][pw].index.to_numpy())),
                                                                                    dic_ref[m][pw].columns.to_numpy())),
                                                                           orient='index').T),
                                               list(dic_ref[m].keys()))))), name_dyes))
        df_ref_int = dint
        for m in name_dyes:
            df_ref_int[m].index = df_ref_int[m].index.levels[0]

    return df_ref_int


def integral_generic_function_v2(file_generic, arg_fit, prefactors, name_dyes, name_single):

    f2 = h5py.File(file_generic.split('.')[0] + '_report.hdf5', 'r')
    df = dict()
    for m in name_single:
        if m in f2['result'].keys():
            d = pd.DataFrame(f2['result'][m].value).T.set_index(0)
        else:
            raise ValueError(m + ' not in report file!')
        df[m] = d

    dict_bFit = dict(map(lambda m: (m, dict(map(lambda k: (k, f2['best values'][m][k].value),
                                                f2['best values'][m].keys()))), name_single))
    f2.close()

    # integral of generic function x pre-factor
    df_integral = pd.DataFrame(np.zeros(shape=(0, 2)), columns=['Integral', 'abs error'])
    for s in name_single:
        bFit = dict_bFit[s]
        if 'amp3' in bFit.keys():
            def _2Voigt_gauss_expl(x):
                c1 = cC._1Voigt_v2(x, weight=bFit['weightV1'], ampG=bFit['ampG1'], cenG=bFit['cenG1'],
                                   widG=bFit['widG1'],
                                   ampL=bFit['ampL1'], cenL=bFit['cenL1'], widL=bFit['widL1'])
                c2 = cC._1Voigt_v2(x, weight=bFit['weightV2'], ampG=bFit['ampG2'], cenG=bFit['cenG2'],
                                   widG=bFit['widG2'],
                                   ampL=bFit['ampL2'], cenL=bFit['cenL2'], widL=bFit['widL2'])
                c3 = cC._1gaussian_v1(x, amp=bFit['amp3'], cen=bFit['cen3'], wid=bFit['wid3'])
                return bFit['weight1'] * c1 + bFit['weight2'] * c2 + bFit['weight3'] * c3

            integrand = _2Voigt_gauss_expl
        else:
            def _Voigt_gauss_expl(x):
                c1 = cC._1Voigt_v2(x, weight=bFit['weightV1'], ampG=bFit['ampG1'], cenG=bFit['cenG1'],
                                   widG=bFit['widG1'],
                                   ampL=bFit['ampL1'], cenL=bFit['cenL1'], widL=bFit['widL1'])
                c2 = cC._1gaussian_v1(x, amp=bFit['amp2'], cen=bFit['cen2'], wid=bFit['wid2'])
                return bFit['weight1'] * c1 + bFit['weight2'] * c2

            integrand = _Voigt_gauss_expl

        integral = integrate.quad(func=integrand, a=arg_fit['fit range ref'][0], b=arg_fit['fit range ref'][1])
        df_integral.loc[s] = integral

    dic_integral = dict(map(lambda m: (m, pd.concat([df_integral['Integral'][name_single[0]] * prefactors[m]['a'],
                                                     df_integral['Integral'][name_single[1]] * prefactors[m]['b']],
                                                    axis=1, keys=name_single)), name_dyes))

    return dic_integral


def integral_generic_function_meas_v3(file_generic, arg_fit, prefactors, name_dyes, name_single):
    # load calibration files and import fitting parameters for reference region using a 2Voigt-gaussian combination
    f2 = h5py.File(file_generic.split('.')[0] + '_report.hdf5', 'r')
    df = dict()
    for m in name_single:
        if m in f2['result'].keys():
            d = pd.DataFrame(f2['result'][m].value).T.set_index(0)
        else:
            raise ValueError(m + ' not in report file!')
        df[m] = d

    dict_bFit = dict(map(lambda m: (m, dict(map(lambda k: (k, f2['best values'][m][k].value),
                                                f2['best values'][m].keys()))), name_single))
    f2.close()

    # integral of generic function x pre-factor
    df_integral = pd.DataFrame(np.zeros(shape=(0, 2)), columns=['Integral', 'abs error'])
    for s in name_single:
        bFit = dict_bFit[s]

        # integral function as it is stored in the calibration file
        def _2Voigt_gauss_expl(x):
            c1 = cC._1Voigt_v2(x, weight=bFit['weightV1'], ampG=bFit['ampG1'], cenG=bFit['cenG1'], widG=bFit['widG1'],
                               ampL=bFit['ampL1'], cenL=bFit['cenL1'], widL=bFit['widL1'])
            c2 = cC._1Voigt_v2(x, weight=bFit['weightV2'], ampG=bFit['ampG2'], cenG=bFit['cenG2'], widG=bFit['widG2'],
                               ampL=bFit['ampL2'], cenL=bFit['cenL2'], widL=bFit['widL2'])
            c3 = cC._1gaussian_v1(x, amp=bFit['amp3'], cen=bFit['cen3'], wid=bFit['wid3'])
            return bFit['weight1'] * c1 + bFit['weight2'] * c2 + bFit['weight3'] * c3

        integrand = _2Voigt_gauss_expl
        integral = integrate.quad(func=integrand, a=arg_fit['fit range ref'][0], b=arg_fit['fit range ref'][1])

        df_integral.loc[s] = integral

    # calculate product of integral and constant prefactor for each single indicator and each pixel
    dic_int = dict(map(lambda m:
                       (m, dict(map(lambda pw:
                                    (pw, pd.concat([prefactors[m].loc[pw].loc['a'] * df_integral['Integral'][name_single[0]],
                                                    prefactors[m].loc[pw].loc['b'] * df_integral['Integral'][name_single[1]]],
                                                   axis=1)), prefactors[m].index.levels[0]))), name_dyes))

    # convert structure of the dictionary: key = dyes; then Dataframe with columns = cube height; index is a
    # multiindex with 1st level cube width and 2nd level prefactors 'a' and 'b'
    dic_integral = dict(map(lambda m: (m, pd.concat(dic_int[m], axis=1).T), name_dyes))

    return dic_integral


# ====================================================================================================================
def preparation_LC(dic_min, name_dyes, name_single, df_generic_corr, arg_fit):
    dic_sen0 = dict()
    dic_generic = dict()

    # check each concentration
    for c in dic_min.keys():
        dic_ = dict()
        # iterate through each dye
        for m in name_dyes:
            # crop the calibration data to the same wavelength as the generic function
            dic_[m] = dic_min[c][m].loc[df_generic_corr.index[0]: df_generic_corr.index[-1]]

            # check for only one pixel -> transfer info to other pixels
            # only once
            ind_new_meas = [round(i, 4) for i in dic_[m].index]
            ind_new_gen = [round(i, 4) for i in df_generic_corr.index]
            dic_[m].index = ind_new_meas
            df_generic_corr.index = ind_new_gen
            dic_generic[m] = df_generic_corr

            if (ind_new_meas == ind_new_gen) is True:
                pass
            else:
                raise ValueError('Wavelength interpolation required to fit measurement and generic function')
        dic_sen0[c] = dic_

    # LC without curve fitting
    lc = arg_fit['range lc']

    # generic function
    dic_sig1 = dict(map(lambda m: (m, dic_generic[m].filter(like=name_single[0]).loc[lc[0]:lc[1]]), name_dyes))
    dic_sig2 = dict(map(lambda m: (m, dic_generic[m].filter(like=name_single[1]).loc[lc[0]:lc[1]]), name_dyes))

    # dictionary for all calibrations
    dic_sens = dict(map(lambda c: (c, dict(map(lambda m: (m, dic_min[c][m].loc[df_generic_corr.index[0]:
                                                                               df_generic_corr.index[-1]].loc[lc[0]:lc[1]]),
                                               name_dyes))), dic_min.keys()))

    return dic_sig1, dic_sig2, dic_sens


def preparation_LC_meas_v1(dic_min, df_generic, name_dyes, name_single, arg_fit):

    dic_generic = dict()
    dic_sen0 = dict()
    # repeat for each dye
    for m in name_dyes:
        px_w = list(dic_min[m].keys())
        # crop the measurement data to the same wavelength as the generic function
        dic_ = dict(map(lambda pw: (pw, dic_min[m][pw].loc[df_generic.index[0]:df_generic.index[-1]]), px_w))

        # check for only one pixel row -> transfer info to other pixels
        ind_new_meas = [round(i, 4) for i in dic_[px_w[0]].index]
        ind_new_gen = [round(i, 4) for i in df_generic.index]
        dic_[px_w[0]].index = ind_new_meas
        df_generic.index = ind_new_gen
        dic_generic[m] = df_generic

        if ([int(i) for i in ind_new_meas] == [int(i) for i in ind_new_gen]) is True:
            pass
        else:
            if len([int(i) for i in ind_new_meas]) == len([int(i) for i in ind_new_gen]):
                df_generic.index = ind_new_meas
                dic_generic[m] = df_generic
            # raise ValueError('Wavelength interpolation required to fit measurement and generic function')
        dic_sen0[m] = dic_

    # LC without curve fitting
    lc = arg_fit['range lc']

    # generic function
    dic_sig1 = dict(map(lambda m: (m, dic_generic[m].filter(like=name_single[0]).loc[lc[0]:lc[1]]), name_dyes))
    dic_sig2 = dict(map(lambda m: (m, dic_generic[m].filter(like=name_single[1]).loc[lc[0]:lc[1]]), name_dyes))

    dic_sens = dict(map(lambda m: (m, dict(map(lambda px: (px, dic_min[m][px].loc[lc[0]:lc[1]]), dic_min[m].keys()))),
                        name_dyes))
    return dic_sig1, dic_sig2, dic_sens


def post_processing(dic_o2_calc, name_dyes, name_single, value_check=True):
    # find position (x, y) of nan and use average of surrounding pixel values for all single indicators
    dic_calc = dict()
    for m in name_dyes:
        dic_calc_single2 = dict()
        for s in name_single:
            df = dic_o2_calc[m][s].copy()

            if value_check is True:
                # check whether the dataframe has nan values
                if df.isnull().any().any() == True:
                    # find position of nan values
                    dic_index = dict(map(lambda c: (c, df[df[c].isnull()].index.tolist()), df.columns))

                    # go through each row of each column
                    df = mean_values_around_nan(dic_index, df)
            dic_calc_single2[s] = df.T
        dic_calc[m] = dic_calc_single2

    # -----------------------------------------------------------------------------------------------
    # combine RoI to one dataframe for each single indicator
    dict_combined = dict(map(lambda s: (s, pd.concat([dic_calc[r][s] for r in name_dyes], axis=0,
                                                     sort=True).sort_index()), name_single))
    for s in name_single:
        dict_combined[s] = dict_combined[s].groupby(by=dict_combined[s].index).mean()

    return dic_calc, dict_combined


def _linear_unmixing(dic_total, dic_known, par0, method='SLSQP', bnds=None):
    # same index for overlaid signal and reference spectra
    doverlay_crop = pd.DataFrame(dic_total.loc[dic_known[0].index[0]:dic_known[0].index[-1]])
    if (doverlay_crop.index == dic_known[0].index).all():
        ydata = doverlay_crop[doverlay_crop.columns[0]].to_numpy()
    else:
        raise ValueError('Adjustment of the index required')

    # same index for both reference spectra
    ar_sig = np.array([dic_known[0][dic_known[0].columns[0]].to_numpy(),
                       dic_known[1][dic_known[1].columns[0]].to_numpy()])

    def func(par):
        dtheory = np.dot(np.array([par[0], par[1]]), ar_sig)
        chi = np.sqrt((ydata - dtheory) ** 2 / (len(ydata) - 2))
        return np.abs(chi).max()
    result = minimize(fun=func, x0=par0, method=method, bounds=bnds)

    return result, doverlay_crop


def linear_unmixing(dic_min, dic_known, par0, ar_sig, arg, method='SLSQP', bnds=None, c=None, plotting=False):
    result, doverlay_crop = _linear_unmixing(dic_total=dic_min, dic_known=dic_known, par0=par0, method=method,
                                             bnds=bnds)

    if plotting is True:
        if c is None:
            title = '$O_2$ (air)'
        else:
            title = '{:.0f}% $O_2$ (air)'.format(c)
        plot.plot_fitresults_lc(doverlay_crop=doverlay_crop, result=result, ar_sig=ar_sig, arg=arg, title=title)

    return result, doverlay_crop


def linear_unmixing_exe(dic_df, arg, bnds, dic_sig1, dic_sig2, lunmix_method, name_ind):
    out = dict(map(lambda m:
                   (m, dict(map(lambda c:
                                (c, linear_unmixing(dic_min=dic_df[c][m], arg=arg, par0=[0.1, 0.1], bnds=bnds,
                                                    dic_known=(dic_sig1[m], dic_sig2[m]), method=lunmix_method,
                                                    ar_sig=[dic_sig1[m], dic_sig2[m]])), sorted(dic_df.keys())))),
                   name_ind))

    results = dict(map(lambda m: (m, [out[m][c][0].x for c in sorted(dic_df.keys())]), name_ind))
    prefactors = dict(map(lambda m: (m, pd.DataFrame(results[m], columns=['a', 'b'], index=sorted(dic_df.keys()))),
                          name_ind))

    doverlay = dict(map(lambda m: (m, pd.concat([out[m][c][1] for c in sorted(dic_df.keys())], axis=1,
                                                keys=sorted(dic_df.keys()))), name_ind))

    return prefactors, results, doverlay


# ====================================================================================================================
def dualSensor_calibration_v2(file_generic, path_dual, path_res, path_ref, arg, arg_fit, name_dyes, name_ind, unit,
                              path_calib=None, val_name=None, ls_outlier=None, name_single=None, ratiometric=True,
                              plot_validation=True, pixel_rot=None, plotting_fit=True, save_res=False, save_op=None,
                              bnds=None, par0=None, corrected_cube=True, save_cube=False, ref_equal=False, threshold=3.,
                              lunmix_method='SLSQP', analyte='O2', simply=True):
    """
    Version 2 - linear combination is replaced by linear unmixing but still it is not yet parallelized. Deviation to
    version 1 starts after preparation_LC(). Be aware that the arg_fit['range lc'] must set to the wavelength range of
    the sensor only.
    :param file_generic:
    :param path_dual:           directory to calibration files which are stored as hdf5-files
    :param path_ref:
    :param path_res:
    :param arg:
    :param arg_fit:             range lc - crop the region for linear unmixing
    :param name_dyes:
    :param val_name:
    :param name_single:
    :param ratiometric:
    :param plot_validation:
    :param plotting_fit:
    :param save_res:
    :param save_op:
    :param bnds:
    :param path_calib:
    :param path_corr:
    :param pixel_90:
    :param corrected_cube:  does path_dual contain already corrected cubes or not? If not provide the path path_calib
                            to the raw cubes so the correction can be conducted
    :param save_cube:
    :param threshold:       threshold for inspection of the spectral deviation of the theoretical sum (fitted) to
                            measured data. Float given in per mil (‰).
    :param lunmix_method:   method chosen for the numpy optimization (minimize) problem.
    :return:
    """

    # load generic function for single indicators
    df_generic_corr = _load_generic_function(file_generic=file_generic, name_single=name_single)

    # -------------------------------------
    # load all corrected calibration cubes from hdf5-files when cube doesn't exist or load cubes and correct them
    dic_calib, dic_calib_STD = _load_calibration(path_dual=path_dual, arg_fit=arg_fit, pixel_rot=pixel_rot, unit=unit,
                                                 corrected_cube=corrected_cube, path_calib=path_calib, analyte=analyte,
                                                 name_dyes=name_dyes, save_cube=save_cube)

    # -------------------------------------
    # real concentration measured with reference sensor
    if path_ref is None:
        ls_conc = dic_calib.keys().tolist()
    else:
        ls_conc = pd.read_csv(path_ref, sep='\t', usecols=[1, 2], index_col=0).sort_index()
    ls_conc = ls_conc.loc[arg_fit['fit concentration'][0]: arg_fit['fit concentration'][1]]

    # ----------------------------------------------------------------------
    # remove outlier and prepare dictionary concentration (in case average replications of concentration points)
    dic_calib, dic_calib_STD = adjust_outlier_concentration_dic(dic_min=dic_calib, dic_std=dic_calib_STD, unit=unit,
                                                                ls_conc=ls_conc, name_dyes=name_dyes,
                                                                ls_outlier=ls_outlier)

    # averaging RoI and standard deviation for error propagation
    dic_av = dict(map(lambda c: (c, dict(map(lambda m: (m, pd.concat(dic_calib[c], axis=1).mean(axis=1)), name_ind))),
                      dic_calib.keys()))
    dic_std = dict(map(lambda c: (c, dict(map(lambda m: (m, pd.concat(dic_calib_STD[c], axis=1).mean(axis=1)),
                                              name_ind))), dic_calib.keys()))

    # ----------------------------------------------------------------------
    # preparation LC without curve fitting -> interpolate wavelength of generic function
    dic_sig1, dic_sig2, dic_sens = preparation_LC(dic_min=dic_av, name_dyes=name_ind, name_single=name_single,
                                                  arg_fit=arg_fit, df_generic_corr=df_generic_corr)

    # ----------------------------------------------------------------------
    # Linear combination - linear unmixing
    [prefactors, results,
     doverlay] = linear_unmixing_exe(dic_df=dic_av, arg=arg, bnds=bnds, dic_sig1=dic_sig1, dic_sig2=dic_sig2,
                                     lunmix_method=lunmix_method, name_ind=name_ind)

    # respective standard deviation
    [prefactors_std, results_std,
     doverlay_std] = linear_unmixing_exe(dic_df=dic_std, arg=arg, bnds=bnds, dic_sig1=dic_sig1, dic_sig2=dic_sig2,
                                         lunmix_method=lunmix_method, name_ind=name_ind)

    # ----------------------------------------------------------------------
    # integration of generic function
    dic_integral = integral_generic_function_v2(file_generic=file_generic, arg_fit=arg_fit, prefactors=prefactors,
                                                name_dyes=name_ind, name_single=name_single)

    # integration of reference region if requested
    if ratiometric is True:
         df_ref_int = integral_reference(dic_data=dic_av, arg_fit=arg_fit, name_dyes=name_ind, ref_equal=ref_equal)
    else:
        df_ref_int = None

    # --------------------------------------------------------------------
    # Stern-Volmer Fit
    dic_SVFit = SternVolmerFit(dic_integral=dic_integral, df_ref_int=df_ref_int, name_dyes=name_ind, par0=par0,
                               name_single=name_single, ratiometric=ratiometric, ref_equal=ref_equal, simply=simply)

    # --------------------------------------------------------------------
    # Plotting
    if plotting_fit is True:
        fig_fit = plot.plotting_fitresultsdual(dic_SVFit=dic_SVFit, dic_raw=dic_SVFit['data'], name_dyes=name_ind,
                                               arg=arg, ratiometric=ratiometric, prefactors_std=prefactors_std,
                                               analyte=analyte, unit=unit, simply=simply)
    else:
        fig_fit = None

    if plot_validation is True:
        if val_name is None:
            m = random.choices(name_dyes)[0]
        else:
            m = val_name

        fig_val = plot.plot_fitresults_lc_all(data=doverlay[m], result=results[m], ar_sig=[dic_sig1[m], dic_sig2[m]],
                                              arg=arg, threshold=threshold, unit=unit)
    else:
        fig_val = None

    # --------------------------------------------------------------------
    # Saving
    if save_res is True:
        save_calibration_dualIndicator(path_dual=path_res, dic_min=dic_av, param_lc=prefactors, fig_val=fig_val,
                                       dic_integral=dic_integral, dic_SVFit=dic_SVFit, ratiometric=ratiometric,
                                       plot_validation=plot_validation, plotting_fit=plotting_fit, fig_fit=fig_fit,
                                       save_op=save_op, standard_dev=prefactors_std, simply=simply)

    return dic_av, prefactors, prefactors_std, dic_integral, df_ref_int, dic_SVFit


def dualSensor_evaluation(file_meas, path_calib, path_res, file_generic, name_dyes, name_single, pixel_rot, arg,
                          arg_fit, max_calib, bnds=None, ref_equal=False, ratiometric=True, plotting=True, saving=False,
                          save_op=None, save_RoI=False, lunmix_method='SLSQP', unit='%air', analyte='O2', cutoff=20,
                          simply=True, value_check=True):
    """

    :param file_meas:
    :param path_calib:
    :param path_res:
    :param file_generic:
    :param name_dyes:
    :param name_single:
    :param pixel_rot:
    :param arg:
    :param arg_fit:
    :param max_calib:       boolean, float or integer; if it is a booloean, it states whether the color bar should be
                            oriented towards the calibration range or the maximal value obtained. Using floats or
                            integers a maximal value can be set
    :param bnds:
    :param ref_equal:
    :param ratiometric:
    :param plotting:
    :param saving:
    :param save_op:
    :param save_RoI:
    :param lunmix_method:
    :param unit:
    :param analyte:
    :param cutoff:
    :param simply:
    :param value_check:     boolean; if True results outside of the calibration range are first set to NaN and then
                            balanced by surrounding pixel values
    :return:
    """
    # load generic function for single indicators
    df_generic_corr = _load_generic_function(file_generic=file_generic, name_single=name_single)

    # -----------------------------------------------------------------------------------------------
    # load calibration and prepare calibration plot
    dic_calib = _load_calibration_dual(path_calib=path_calib, ratiometric=ratiometric, simply=simply)

    # -----------------------------------------------------------------------------------------------
    # check whether the sensor is the same as the calibration file
    for m in name_dyes:
        if m.split(' ')[0] in dic_calib['Fit']['best Fit'].keys():
            pass
        else:
            raise ValueError('Could not find the sensor {} in the calibration file. Only '.format(m),
                             dic_calib['bestFit'].keys(), ' are available')

    # -----------------------------------------------------------------------------------------------
    # correction of cube and select regions of interest
    cube, fig, ax = corr.hyperCube_preparation(file_hdr=file_meas, plotting=False, arg=arg, name_dyes=name_dyes,
                                               pixel_rot=pixel_rot, save=save_RoI, averaging=False, unit=unit,
                                               analyte=analyte, cube_type='multiple')

    # rearrangement of cube
    dic_meas = arrange_data(cube['region of interest'], name_dyes=name_dyes, fr=None)

    # ----------------------------------------------------------------------
    # preparation LC without curve fitting -> interpolate wavelength of generic function
    dic_sig1, dic_sig2, dic_sens = preparation_LC_meas_v1(dic_min=dic_meas, df_generic=df_generic_corr, arg_fit=arg_fit,
                                                          name_dyes=name_dyes, name_single=name_single)

    # ----------------------------------------------------------------------
    # Linear combination - linear unmixing in each pixel of the measurement cube
    out = dict(map(lambda m: (m, dict(map(lambda pw:
                                          (pw, dict(map(lambda ph:
                                                        (ph, linear_unmixing(dic_min=dic_meas[m][pw][ph], arg=arg,
                                                                             bnds=bnds, par0=[0.1, 0.1],
                                                                             method=lunmix_method,
                                                                             dic_known=(dic_sig1[m], dic_sig2[m]),
                                                                             ar_sig=[dic_sig1[m], dic_sig2[m]])),
                                                        list(dic_meas[m][pw].keys())))), list(dic_meas[m].keys())))),
                   name_dyes))

    prefactors = dict(map(lambda m:
                          (m, pd.concat(dict(map(lambda pw: (pw, pd.DataFrame([out[m][pw][ph][0].x
                                                                               for ph in list(out[m][pw].keys())],
                                                                              index=list(out[m][pw].keys()),
                                                                              columns=['a', 'b']).T),
                                                 list(out[m].keys()))))), name_dyes))

    # ----------------------------------------------------------------------
    # integration of sensor region
    dic_integral = integral_generic_function_meas_v3(file_generic=file_generic, arg_fit=arg_fit, prefactors=prefactors,
                                                     name_dyes=name_dyes, name_single=name_single)

    # integration of reference region if requested
    if ratiometric is True:
         df_ref_int = integral_reference_pixel(dic_data=dic_meas, arg_fit=arg_fit, name_dyes=name_dyes,
                                               ref_equal=ref_equal)
    else:
        df_ref_int = None

    # ----------------------------------------------------------------------
    # define ratio of the integrals indicator / reference
    if ratiometric is True:
        if ref_equal is False:
            dic_iratio = dict(map(lambda m:
                                  (m, pd.concat(dict(map(lambda pw: (pw, dic_integral[m].loc[pw] / df_ref_int[m].loc[pw]),
                                                         dic_integral[m].index.levels[0])))), name_dyes))
        else:
            dic_iratio = dict(map(lambda m: (m, dic_integral[m] / df_ref_int.loc[m, 0]), name_dyes))
    else:
        dic_iratio = dic_integral

    # determine analyte concentration using simplified SVfit
    dic_iratio, dic_o2_calc = analyte_concentration_SVFit_pixel(dic_iratio=dic_iratio, dic_calib=dic_calib,
                                                                name_dyes=name_dyes, name_single=name_single,
                                                                cutoff=cutoff, simply=simply, value_check=value_check)

    # -----------------------------------------------------------------------------------------------
    # post-processing of analyte determination
    dic_calc2, dict_combined = post_processing(dic_o2_calc=dic_o2_calc, name_dyes=name_dyes, name_single=name_single,
                                               value_check=value_check)

    # -----------------------------------------------------------------------------------------------
    # plotting results
    # create image frame with all values zero besides the one from the RoI (dictionary - keys are single indicators)
    dict_frame = dict()
    for s in name_single:
        frame = pd.DataFrame(np.zeros(shape=cube['Cube']['cube'].shape[:-1])) # original orientation of cube
        frame.loc[dict_combined[s].index[0]:dict_combined[s].index[-1],
        dict_combined[s].columns[0]:dict_combined[s].columns[-1]] = dict_combined[s]
        dict_frame[s] = frame

    dfig_im, dfigures = plot.plotting_dualIndicator(file_meas=file_meas, meas_dyes=name_dyes, name_single=name_single,
                                                    cube_corr=cube, dic_o2_calc=dic_calc2, plotting=plotting, arg=arg,
                                                    unit=unit, analyte=analyte, frame=dict_frame, cmap=arg['cmap'],
                                                    dic_calib=dic_calib, simply=simply, max_calib=max_calib,
                                                    cutoff=cutoff)

    # -------------------------------------------------------------------------------------------------------------
    # saving
    if saving is True:
        # preparation of data set for export
        # header
        dic_metadata = dict({'boundaries linear unmixing': str(bnds), 'calibration file': path_calib,
                             'concentration': cube['Concentration'], 'fitting linear unmixing': arg_fit,
                             'generic function': file_generic, 'measurement': file_meas.split('\\')[-1],
                             'ratiometric': str(ratiometric), 'sensor ID': name_dyes, 'wavelength': cube['wavelength']})

        # data
        # re-transform 90deg rotation
        dic_iratio_ = dic_iratio.copy()
        for s in name_single:
            for em, m in enumerate(name_dyes):
                xorigin = [np.int(i)
                           for i in np.linspace(pixel_rot[em][1][0], pixel_rot[em][0][0],
                                                num=int((pixel_rot[em][1][0] - pixel_rot[em][0][0]) / 1 + 1))]
                if np.all(sorted(xorigin) == dic_iratio_[s][m].index) == True:
                    pass
                else:
                    # otherwise transform
                    xnew = [cube['Cube']['cube'].shape[1] - d for d in dic_iratio_[s][m].index.levels[0]]
                    dic_iratio_[s][m].index = xnew
                dic_iratio_[s][m] = dic_iratio_[s][m].sort_index()

        # combine RoI for each single indicator and re-convert index
        dic_iratio_comb = dict(map(lambda s: (s, pd.concat([dic_iratio_[s][n] for n in name_dyes], axis=0,
                                                           sort=True).sort_index()), name_single))

        # combine relevant calibration data
        dict_calib = dict()
        dict_calib['Fit calib data'] = pd.concat([pd.DataFrame.from_dict(dic_calib['Fit']['data'][m],
                                                                         orient='index').T for m in name_dyes])
        dict_calib['integral calib'] = pd.concat([dic_calib['data']['integral'][m] for m in name_dyes])
        dict_calib['calib points'] = list(dict_calib['Fit calib data'].index)

        dic_data = dict({'pixel of interest': cube['pixel of interest'], 'iratio': dic_iratio_comb,
                         'raw data RoI': cube['region of interest'], 'calibration data': dict_calib})

        # results
        dic_res = dict(map(lambda m: (m, dic_calc2[m]), name_dyes))

        save_measurement_dualIndicator(file_meas=file_meas, dic_metadata=dic_metadata, dic_data=dic_data,
                                       dic_res=dic_res, ratiometric=ratiometric, dfig_im=dfig_im, save_op=save_op,
                                       dic_figures=dfigures, path_res=path_res, simply=simply)

    return cube, prefactors, dic_iratio, dic_calib, dic_calc2, dfig_im, dfigures, dic_iratio_comb
