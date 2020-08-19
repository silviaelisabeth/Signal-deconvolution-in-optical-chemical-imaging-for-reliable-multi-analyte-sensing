__author__ = 'Silvia E Zieger'
__project__ = 'multi-analyte imaging using hyperspectral camera systems'

"""Copyright 2020. All rights reserved.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable 
for any damages arising from the use of this software.
Permission is granted to anyone to use this software within the scope of evaluating mutli-analyte sensing. No permission
is granted to use the software for commercial applications, and alter it or redistribute it.

This notice may not be removed or altered from any distribution.
"""

import numpy as np
import pandas as pd
from lmfit import Model
from lmfit.models import VoigtModel
from scipy.optimize import minimize
import time
from glob import glob
import os.path
import pathlib
import h5py

import correction_hyperCamera_v1_4 as corr
import layout_plotting_v1_3 as plot


# ====================================================================================================================
def _1gaussian_v1(x, amp, cen, wid):
    return amp * np.exp(-1*(np.log(2))*((cen-x)*2/wid)**2)


def _1Voigt_v1(x, weightG, weightL, ampG, cenG, widG, ampL, cenL, widL):
    return weightG*(ampG * np.exp(-1*(np.log(2))*((cenG-x)*2/widG)**2)) +\
           weightL*(ampL / (1 + ((cenL-x)*2/widL)**2))


def _1Voigt_v2(x, weight, ampG, cenG, widG, ampL, cenL, widL):
    return weight*(ampG * np.exp(-1*(np.log(2))*((cenG-x)*2/widG)**2)) +\
           (1-weight)*(ampL / (1 + ((cenL-x)*2/widL)**2))


def _Voigt_gauss_v1(x, weightV1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, amp2, cen2, wid2, weight1, weight2):
    c1 = _1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1, widL=widL1)
    c2 = _1gaussian_v1(x, amp=amp2, cen=cen2, wid=wid2)
    return weight1*c1 + weight2*c2


def _2Voigt_gauss_v1(x, weightV1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, weightV2, ampG2, cenG2, widG2, ampL2, cenL2,
                     widL2, amp3, cen3, wid3, weight1, weight2, weight3):
    c1 = _1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1, widL=widL1)
    c2 = _1Voigt_v2(x, weight=weightV2, ampG=ampG2, cenG=cenG2, widG=widG2, ampL=ampL2, cenL=cenL2, widL=widL2)
    c3 = _1gaussian_v1(x, amp=amp3, cen=cen3, wid=wid3)
    return weight1*c1 + weight2*c2 + weight3*c3


def optimize_2peaks_v2(df_sensor, df_rest):
    # noise + Sensor
    model2 = Model(_Voigt_gauss_v1)

    # parameter preparation
    widG1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widL1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    wid2 = (df_rest.index[-1] - df_rest.index[0])
    par_ = dict({'ampG1': df_sensor.loc[:widG1 + df_sensor.index[0]].max(),
                 'cenG1': df_sensor.loc[:widG1 + df_sensor.index[0]].idxmax(),
                 'ampL1': df_sensor.loc[widG1 + df_sensor.index[0] + 30:].max(),
                 'cenL1': df_sensor.loc[widG1 + df_sensor.index[0] + 30:].idxmax(),
                 'amp2': df_rest.loc[:wid2 + df_rest.index[0]].max(),
                 'cen2': df_rest.loc[:wid2 + df_rest.index[0]].idxmax()})
    params2 = model2.make_params(weightV1=0.5, weight1=1., weight2=0.5, ampG1=par_['ampG1'], cenG1=par_['cenG1'],
                                 widG1=widG1, ampL1=par_['ampL1'], cenL1=par_['cenL1'], widL1=widL1,  wid2=wid2,
                                 amp2=par_['amp2'], cen2=par_['cen2'])

    # parameter boundaries
    params2['widG1'].min = 0
    params2['widL1'].min = 0
    params2['ampG1'].min = 0
    params2['ampL1'].min = 0
    params2['cenG1'].min = 0
    params2['cenL1'].min = 0
    params2['amp2'].min = 0
    params2['cen2'].min = 0
    params2['wid2'].min = 0
    params2['weight1'].min = 0
    params2['weight2'].min = 0

    # ==========================================
    # lmfit - fitting
    df_tofit = pd.concat([pd.DataFrame(df_sensor), pd.DataFrame(df_rest)], axis=0).mean(axis=1)
    df_tofit = df_tofit.sort_index()
    result_ = model2.fit(df_tofit.values.tolist(), params2, x=np.array(df_tofit.index), nan_policy='omit')
    df_result = pd.DataFrame(result_.best_fit, index=df_tofit.index)

    return df_result, result_, df_tofit


def optimize_3peaks_v2(df_sensor, df_rest, df_ref, arg_fit):
    # noise + Sensor + reference
    model3 = Model(_2Voigt_gauss_v1)

    # parameter preparation
    widG1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widL1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widG2 = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2
    wid3 = (df_rest.index[-1] - df_rest.index[0])

    par_ = dict({'ampG1': df_sensor.loc[:widG1 + df_sensor.index[0]].max(),
                 'cenG1': df_sensor.loc[:widG1 + df_sensor.index[0]].idxmax(),
                 'ampL1': df_sensor.loc[widG1 + df_sensor.index[0] + 30:].max(),
                 'cenL1': df_sensor.loc[widG1 + df_sensor.index[0] + 30:].idxmax(),
                 'ampL2': df_ref.loc[arg_fit['fit range ref'][0] + widG2:arg_fit['fit range ref'][1]].max(),
                 'ampG2': df_ref.max(), 'cenG2': df_ref.idxmax(),
                 'cenL2': df_ref.loc[arg_fit['fit range ref'][0] + widG2:arg_fit['fit range ref'][1]].idxmax(),
                 'widL2': (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2,
                 'amp3': df_rest.loc[:wid3 + df_rest.index[0]].max(),
                 'cen3': df_rest.loc[:wid3 + df_rest.index[0]].idxmax()})

    params3 = model3.make_params(weightV1=0.5,  weightV2=0.5, weight1=1., weight2=0.5, weight3=0.5, ampG1=par_['ampG1'],
                                 widL1=widL1, widG1=widG1, wid3=wid3,  widG2=widG2, cenG1=par_['cenG1'],
                                 ampL1=par_['ampL1'], cenL1=par_['cenL1'], ampG2=par_['ampG2'], cenG2=par_['cenG2'],
                                 ampL2=par_['ampL2'], cenL2=par_['cenL2'], widL2=par_['widL2'], amp3=par_['amp3'],
                                 cen3=par_['cen3'])

    # define the frame of the sensor curves
    params3['weightV1'].min = 0.3
    params3['weightV1'].max = 0.6
    params3['widG1'].min = 0
    params3['widL1'].min = 0
    params3['widL1'].max = 100
    params3['ampG1'].min = 0
    params3['ampG1'].max = 10
    params3['ampL1'].min = 0
    params3['ampL1'].max = 10
    params3['cenG1'].min = 0
    params3['cenL1'].min = 0

    params3['ampG2'].min = 0
    params3['ampG2'].max = 10
    params3['ampL2'].min = 0
    params3['ampL2'].max = 10
    params3['cenG2'].min = 0
    params3['cenL2'].min = 0
    params3['widG2'].min = 0
    params3['widL2'].min = 0
    params3['widG2'].max = 100
    params3['widL2'].max = 100

    params3['amp3'].min = 0
    params3['amp3'].max = 10
    params3['cen3'].min = 0
    params3['wid3'].min = 0
    params3['wid3'].max = 100
    params3['weight1'].min = 0
    params3['weight2'].min = 0
    params3['weight3'].min = 0

    # ==========================================
    # lmfit - fitting
    df_tofit = pd.concat([pd.DataFrame(df_sensor), pd.DataFrame(df_ref), pd.DataFrame(df_rest)], axis=0).mean(axis=1)
    df_tofit = df_tofit.sort_index()
    result_ = model3.fit(df_tofit.values.tolist(), params3, x=np.array(df_tofit.index), nan_policy='omit')
    df_result = pd.DataFrame(result_.best_fit, index=df_tofit.index)

    return df_result, result_, df_tofit


def curve_fitting_reference(df_ref, arg_fit):
    # store as dictionary or dataframe
    # noise + Sensor + reference
    modelRef = Model(_1Voigt_v2)

    widG = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2
    ampL = df_ref.loc[arg_fit['fit range ref'][0] + widG:arg_fit['fit range ref'][1]].max()
    cenL = df_ref.loc[arg_fit['fit range ref'][0] + widG:arg_fit['fit range ref'][1]].idxmax()
    widL = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2

    paramsRef = modelRef.make_params(weight=0.5, ampG=df_ref.max(), cenG=df_ref.idxmax(), widG=widG, ampL=ampL,
                                     cenL=cenL, widL=widL)
    paramsRef['ampG'].min = 0
    paramsRef['ampG'].max = 10
    paramsRef['ampL'].min = 0
    paramsRef['ampL'].max = 10
    paramsRef['cenG'].min = 0
    paramsRef['cenL'].min = 0
    paramsRef['widG'].min = 0
    paramsRef['widL'].min = 0
    paramsRef['widG'].max = 100
    paramsRef['widL'].max = 100

    # ==========================================
    # lmfit - fitting
    df_tofit = pd.DataFrame(df_ref)
    df_tofit = df_tofit.sort_index()
    result_ = modelRef.fit(df_tofit[df_tofit.columns[0]].values.tolist(), paramsRef, x=np.array(df_tofit.index),
                           nan_policy='omit')
    df_result = pd.DataFrame(result_.best_fit, index=df_tofit.index)

    return df_result, result_


def split_spectrum2parts(df_sens1, df_sens2, ls_sensor_fit, what_fit, arg_fit):
    # define reference region
    if 'reference' in what_fit:
        df_ref = pd.concat([pd.DataFrame(
            df_sens1.filter(like='mean').loc[arg_fit['fit range ref'][0]:arg_fit['fit range ref'][1]], ).astype(float),
                            pd.DataFrame(df_sens2.filter(like='mean').loc[
                                         arg_fit['fit range ref'][0]:arg_fit['fit range ref'][1]], ).astype(float)],
                           axis=1, sort=True)
        df_ref.columns = ls_sensor_fit
    else:
        df_ref = None

    # define background - noise / artifacts
    if 'middle' in what_fit:
        df_rest = pd.concat([pd.DataFrame(
            df_sens1.filter(like='mean').loc[arg_fit['fit range ref'][1]:arg_fit['fit range sensor'][0]]).astype(float),
                             pd.DataFrame(df_sens2.filter(like='mean').loc[
                                          arg_fit['fit range ref'][1]:arg_fit['fit range sensor'][0]]).astype(float)],
                            axis=1, sort=True)
        df_rest.columns = ls_sensor_fit
    else:
        df_rest = None

    # define sensor region
    df_sens = pd.concat([pd.DataFrame(
        df_sens1.filter(like='mean').loc[arg_fit['fit range sensor'][0]: arg_fit['fit range sensor'][1]]).astype(float),
                         pd.DataFrame(df_sens2.filter(like='mean').loc[
                                      arg_fit['fit range sensor'][0]: arg_fit['fit range sensor'][1]]).astype(float)],
                        axis=1, sort=True)
    df_sens.columns = ls_sensor_fit

    return df_ref, df_rest, df_sens


# ====================================================================================================================
def minimize_pixel_v2(dic, key, par0, ar_sig, ydata, bnds=None):
    def func(par):
        dtheory = np.dot(np.array([par[0], par[1]]), ar_sig)
        chi = np.sqrt((ydata - dtheory) ** 2 / (len(ydata) - 2))
        return np.abs(chi).max()

    result = minimize(fun=func, x0=par0, method='SLSQP', bounds=bnds)
    dic[key] = result.x


def _baseline_corr(d_sub_mp, key, df):
    l = df - df.min()

    d_sub_mp[key] = (l)


def _optimize_3peaks_v2(dic_mp, key, df_sensor, df_rest, df_ref, arg_fit):
    # noise + Sensor + reference
    model3 = Model(_2Voigt_gauss_v1)

    # parameter preparation
    frr = arg_fit['fit range ref']
    widG1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widL1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widG2 = (frr[1] - frr[0]) / 2
    wid3 = (df_rest.index[-1] - df_rest.index[0])

    params3 = model3.make_params(weightV1=0.5,  weightV2=0.5, weight1=1., weight2=0.5, weight3=0.5, widL1=widL1,
                                 widG1=widG1, wid3=wid3,  widG2=widG2, ampG2=df_ref.max(), cenG2=df_ref.idxmax(),
                                 ampG1=df_sensor.loc[:widG1 + df_sensor.index[0]].max(), widL2=(frr[1] - frr[0]) / 2,
                                 cenG1=df_sensor.loc[:widG1 + df_sensor.index[0]].idxmax(),
                                 ampL1=df_sensor.loc[widG1 + df_sensor.index[0] + 30:].max(),
                                 cenL1=df_sensor.loc[widG1 + df_sensor.index[0] + 30:].idxmax(),
                                 ampL2=df_ref.loc[frr[0] + widG2:frr[1]].max(),
                                 cenL2=df_ref.loc[frr[0] + widG2:frr[1]].idxmax(),
                                 amp3=df_rest.loc[:wid3 + df_rest.index[0]].max(),
                                 cen3=df_rest.loc[:wid3 + df_rest.index[0]].idxmax())

    # define the frame of the sensor curves
    params3['weightV1'].min = 0.3
    params3['weightV1'].max = 0.6
    params3['widG1'].min = 0
    params3['widL1'].min = 0
    params3['widL1'].max = 100
    params3['ampG1'].min = 0
    params3['ampG1'].max = 10
    params3['ampL1'].min = 0
    params3['ampL1'].max = 10
    params3['cenG1'].min = 0
    params3['cenL1'].min = 0

    params3['ampG2'].min = 0
    params3['ampG2'].max = 10
    params3['ampL2'].min = 0
    params3['ampL2'].max = 10
    params3['cenG2'].min = 0
    params3['cenL2'].min = 0
    params3['widG2'].min = 0
    params3['widL2'].min = 0
    params3['widG2'].max = 100
    params3['widL2'].max = 100

    params3['amp3'].min = 0
    params3['amp3'].max = 10
    params3['cen3'].min = 0
    params3['wid3'].min = 0
    params3['wid3'].max = 100
    params3['weight1'].min = 0
    params3['weight2'].min = 0
    params3['weight3'].min = 0

    # ==========================================
    # lmfit - fitting
    df_tofit = pd.concat([pd.DataFrame(df_sensor), pd.DataFrame(df_ref), pd.DataFrame(df_rest)], axis=0).mean(axis=1)
    df_tofit = df_tofit.sort_index()
    result_ = model3.fit(df_tofit.values.tolist(), params3, x=np.array(df_tofit.index), nan_policy='omit')

    dic_mp[key] = result_


def _curve_fitting_reference(dic_cf_mp, key, df_ref, arg_fit):
    # store as dictionary or dataframe
    # noise + Sensor + reference
    modelRef = Model(_1Voigt_v2)

    widG = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2
    ampL = df_ref.loc[arg_fit['fit range ref'][0] + widG:arg_fit['fit range ref'][1]].max()
    cenL = df_ref.loc[arg_fit['fit range ref'][0] + widG:arg_fit['fit range ref'][1]].idxmax()
    widL = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2

    paramsRef = modelRef.make_params(weight=0.5, ampG=df_ref.max(), cenG=df_ref.idxmax(), widG=widG, ampL=ampL,
                                     cenL=cenL, widL=widL)
    paramsRef['ampG'].min = 0
    paramsRef['ampG'].max = 10
    paramsRef['ampL'].min = 0
    paramsRef['ampL'].max = 10
    paramsRef['cenG'].min = 0
    paramsRef['cenL'].min = 0
    paramsRef['widG'].min = 0
    paramsRef['widL'].min = 0
    paramsRef['widG'].max = 100
    paramsRef['widL'].max = 100

    # ==========================================
    # lmfit - fitting
    df_tofit = pd.DataFrame(df_ref)
    df_tofit = df_tofit.sort_index()
    result_ = modelRef.fit(df_tofit[df_tofit.columns[0]].values.tolist(), paramsRef, x=np.array(df_tofit.index),
                           nan_policy='omit')

    dic_cf_mp[key] = result_


# ===================================================================================================================
def curve_fitting_voigt(dref, pars=None):
    xdata = dref.index.to_numpy()
    ydata = dref.to_numpy()

    mod = VoigtModel()
    if pars is None:
        pars = mod.guess(ydata, x=xdata)

    out = mod.fit(ydata, pars, x=xdata)

    return out


# ====================================================================================================================
def _linearcombinationDual_v2(data_super, df_sig1, df_sig2, start_, stop_, num=100):
    range1_ = np.linspace(start=start_[0], stop=stop_[0], num=int(num))
    range1 = list(dict.fromkeys(range1_))
    range2_ = np.linspace(start=start_[1], stop=stop_[1], num=int(num))
    range2 = list(dict.fromkeys(range2_))

    dic_residual = dict.fromkeys(set(range1))
    dic_sqr = dict.fromkeys(set(range1))
    for a in range1:
        dic_residual[a] = pd.concat([a*df_sig1 + b*df_sig2 - data_super for b in range2], axis=1)

        df_res = dic_residual[a].dropna()**2
        dic_sqr[a] = np.sqrt(df_res.sum() / (len(dic_residual[a].dropna())-2))

    df_sqr = pd.concat(dic_sqr, axis=1).sort_index(axis=1).T
    df_sqr.columns = range2
    best_lc = df_sqr.min().min()
    best_a = df_sqr.min(axis=1).idxmin()
    best_b = df_sqr.min().idxmin()

    dic_params = dict({'sqr': df_sqr, 'bestFit': best_lc, 'Sensor1': best_a, 'Sensor2': best_b})

    return dic_params


def stochastic_parameter_optimization_pixel(df_sensor, df_sig1, df_sig2, nruns, start_0, stop_0, num=10):
    dic_params_pc = dict.fromkeys(set(np.arange(nruns)))

    for n in range(nruns):
        if n == 0:
            dic_params_pc[n] = _linearcombinationDual_v2(data_super=df_sensor, df_sig1=df_sig1, df_sig2=df_sig2,
                                                         num=num, start_=start_0, stop_=stop_0)
        else:
            step_n = (dic_params_pc[n - 1]['sqr'].index[-1] - dic_params_pc[n - 1]['sqr'].index[0]) / num * 10
            start_n = [dic_params_pc[n - 1]['Sensor1'] - step_n, dic_params_pc[n - 1]['Sensor2'] - step_n]
            stop_n = [dic_params_pc[n - 1]['Sensor1'] + step_n, dic_params_pc[n - 1]['Sensor2'] + step_n]

            dic_params_pc[n] = _linearcombinationDual_v2(data_super=df_sensor, df_sig1=df_sig1, df_sig2=df_sig2,
                                                         num=num, start_=start_n, stop_=stop_n)

    return dic_params_pc


def pixel_LCanalysis(df_sens, df_sig1, df_sig2, nruns, start_0, stop_0, num=50):
    if isinstance(start_0, float):
        start_0 = [start_0] * 2
    elif isinstance(start_0, int):
        start_0 = [np.float(start_0)] * 2
    if isinstance(stop_0, float):
        stop_0 = [stop_0] * 2
    elif isinstance(stop_0, int):
        stop_0 = [np.float(stop_0)] * 2

    dic_params = stochastic_parameter_optimization_pixel(df_sensor=df_sens, df_sig1=df_sig1, df_sig2=df_sig2, num=num,
                                                         nruns=nruns, start_0=start_0, stop_0=stop_0)

    # post-analysis of linear combination
    bestFit = dict(map(lambda r: (r, dic_params[r]['bestFit']), dic_params.keys())) # dict()
    # for r in dic_params.keys():
    #     bestFit[r] = dic_params[r]['bestFit']

    fitofchoice = pd.DataFrame([bestFit]).idxmin(axis=1).values[0]
    dic_lc = dict({'Sensor1': dic_params[fitofchoice]['Sensor1'], 'Sensor2': dic_params[fitofchoice]['Sensor2'],
                   'bestFit': dic_params[fitofchoice]['bestFit']})
    return dic_lc


def compute_lc(dic_sens, start, stop, num, dic_sig1, dic_sig2, nruns):
    l = list((pixel_LCanalysis(df_sens=dic_sens, num=num, df_sig1=dic_sig1, start_0=start, df_sig2=dic_sig2,
                               stop_0=stop, nruns=nruns)).items())

    return np.array([l[0][1], l[1][1], l[2][1]])


def save_genericfunction(path_res, dic_cube, df_sens1, df_sens2, dic_fitting):
    # preparation for saving
    # list of single indicators used
    ind_ls = [c.split('mean')[0].strip() for c in dic_cube[list(dic_cube.keys())[0]].filter(like='mean').columns]
    # date and general file name
    now = str(time.localtime().tm_year) + str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
    save_name = now + '_Generic_function'

    if path_res.endswith('/') == True:
        path_generic = path_res + 'Generic_function/'
        savingname = path_generic + save_name
    else:
        path_generic = path_res + '/' + 'Generic_function/'
        savingname = path_generic + save_name

    # check whether directory already exist
    if not os.path.exists(path_generic):
        pathlib.Path(path_generic).mkdir(parents=True, exist_ok=True)
    else:
        pass

    # check whether file already exist
    if os.path.isfile(savingname + '.txt') == False:
        pass
    else:
        ls_files_exist = glob(savingname + '*.txt')
        f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
        num = 0
        for f in f_exist:
            if 'run' in f:
                num = int(f.split('run')[-1]) + 1
            else:
                pass
        savingname = savingname + '_run' + str(num)

    df_out = pd.concat([df_sens1, df_sens2], axis=1, sort=True)
    df_out.to_csv(savingname + '.txt', sep='\t')

    # transform dictionary into array to store it into an hdf5 File
    savingname = savingname + '_report.hdf5'
    f1 = h5py.File(savingname, "w")

    grp_res = f1.create_group("result")
    for m in ind_ls:
        ar_res0 = np.array([dic_fitting['result'][m].index.to_numpy(), dic_fitting['result'][m].to_numpy()])
        grp_res.create_dataset(m, ar_res0.shape, data=ar_res0)

    grp_bv = f1.create_group("best values")
    for m in ind_ls:
        subgrp_bv = grp_bv.create_group(m)
        for k, v in dic_fitting['report'][m].best_values.items():
            subgrp_bv.create_dataset(str(k), v.shape, data=np.array(v))
    f1.close()


# ====================================================================================================================
def generic_function_cube(path_100, pixel_90, arg, arg_fit, name_dyes, ls_sensor_fit, what_fit, unit, plot_meas=False,
                          plot_result=False, saving=False, path_res=None):
    """
    Correction of the whole cube for all calibration measurement files within the given directory. Then, selection of
    the first 2-3 highest intensities to determine the generic function of the single indicator.
    The function is assigned to fit 2 single indicators at the same time.
    :param path_100:
    :param pixel_90:
    :param arg:
    :param arg_fit:
    :param name_dyes:
    :param what_fit:
    :param path_corr:
    :param averaging:
    :param plot_meas:
    :return:
    """

    # correction of measurement file and split into regions of  - averaging is required
    # use only calibration points that will be used afterwards for averaging
    fit_range = np.arange(arg_fit['fit concentration'][0], arg_fit['fit concentration'][1]+1)

    dic_cube = dict()
    for i in glob(path_100 + '/*.hdr'):
        if np.float(i.split(unit)[0].split('_')[-1]) in fit_range:
            cube_corr, fig, ax = corr.correction_hyperCube(i, arg=arg, name_dyes=name_dyes, save=False,
                                                           pixel_90=pixel_90, averaging=True, plotting=plot_meas)
            dic_cube[cube_corr['Concentration']] = cube_corr['average data']

    if len(list(dic_cube.keys())) < 2:
        print('WARNING!')
        print('The standard deviation of the generic function is empty which can cause problems for your evaluation.')

    # ---------------------------------------------------------------------------------
    # relative spectrum to the maximum in the sensor region
    d_norm = [dic_cube[k].filter(like=' mean') / dic_cube[k].filter(like=' mean').loc[
                                                 arg_fit['fit range sensor'][0]:arg_fit['fit range sensor'][1]].max()
              for k in dic_cube.keys()]

    df_sens1 = pd.concat([pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[0]).mean(axis=1),
                          pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[0]).std(axis=1)], axis=1,
                         keys=[ls_sensor_fit[0] + ' mean', ls_sensor_fit[0] + ' STD'])
    df_sens2 = pd.concat([pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[1]).mean(axis=1),
                          pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[1]).std(axis=1)], axis=1,
                         keys=[ls_sensor_fit[1] + ' mean', ls_sensor_fit[1] + ' STD'])

    # ---------------------------------------------------------------------------------
    # split into regions (ref, sensor, background) for curve fitting
    if 'reference' in what_fit:
        df_ref = pd.concat([pd.DataFrame(
            df_sens1.filter(like='mean').loc[arg_fit['fit range ref'][0]:arg_fit['fit range ref'][1]], ).astype(float),
                            pd.DataFrame(df_sens2.filter(like='mean').loc[
                                         arg_fit['fit range ref'][0]:arg_fit['fit range ref'][1]], ).astype(float)],
                           axis=1, sort=True)
        df_ref.columns = ls_sensor_fit
    else:
        df_ref = None

    if 'middle' in what_fit:
        df_rest = pd.concat([pd.DataFrame(
            df_sens1.filter(like='mean').loc[arg_fit['fit range ref'][1]:arg_fit['fit range sensor'][0]]).astype(float),
                             pd.DataFrame(df_sens2.filter(like='mean').loc[
                                          arg_fit['fit range ref'][1]:arg_fit['fit range sensor'][0]]).astype(float)],
                            axis=1, sort=True)
        df_rest.columns = ls_sensor_fit
    else:
        df_rest = None

    df_sens = pd.concat([pd.DataFrame(
        df_sens1.filter(like='mean').loc[arg_fit['fit range sensor'][0]: arg_fit['fit range sensor'][1]]).astype(float),
                         pd.DataFrame(df_sens2.filter(like='mean').loc[
                                      arg_fit['fit range sensor'][0]: arg_fit['fit range sensor'][1]]).astype(float)],
                        axis=1, sort=True)
    df_sens.columns = ls_sensor_fit

    # ---------------------------------------------------------------------------------
    # Curve fitting of averaged generic function
    if 'reference' in what_fit:
        dic_func = dict(map(lambda d: (d, optimize_3peaks_v2(df_sensor=df_sens[d], df_rest=df_rest[d],
                                                             df_ref=df_ref[d], arg_fit=arg_fit)), ls_sensor_fit))
    else:
        dic_func = dict(map(lambda d: (d, optimize_2peaks_v2(df_sensor=df_sens[d], df_rest=df_rest[d])), ls_sensor_fit))

    # split results and store them into fitting dictionary
    dic_result = pd.concat([dic_func[d][0] for d in ls_sensor_fit], axis=1, keys=ls_sensor_fit)
    dic_report = dict(map(lambda d: (d, dic_func[d][1]), ls_sensor_fit))
    df_sens_full = pd.concat([df_sens, df_sens1.filter(like='STD'), df_sens2.filter(like='STD')], axis=1,
                             sort=True).loc[df_sens.index]
    dic_fitting = dict({'result': dic_result, 'report': dic_report, 'sensor': df_sens_full})

    # -----------------------------------------------------------------------
    # Plotting
    if plot_result is True:
        df_std = pd.concat([df_sens1.filter(like='STD'), df_sens2.filter(like='STD')], axis=1,
                           sort=True).loc[arg_fit['fit range ref'][0]: arg_fit['fit range sensor'][1]]
        df_std.columns = ls_sensor_fit
        df_mean = pd.concat([df_sens1.filter(like='mean'), df_sens2.filter(like='mean')], axis=1,
                            sort=True).loc[arg_fit['fit range ref'][0]: arg_fit['fit range sensor'][1]]
        df_mean.columns = ls_sensor_fit

        fig, ax = plot.plotting_fit_results_2Sensors(df_sensors=df_mean, df_sensors_std=df_std, arg=arg)
        # fig, ax = plot.plotting_fit_results_2Sensors(df_sensors=df_sensors, df_sensors_std=df_sensors_std, arg=arg)
    else:
        fig = None
        ax = None

    # -----------------------------------------------------------------------
    # Saving
    if saving is True:
        if path_res is None:
            print('Directory to save the output is required. Please provide path_res')

        now = str(time.localtime().tm_year) + str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
        save_name = now + '_Generic_function'

        if path_res.endswith('/') == True:
            path_generic = path_res + 'Generic_function/'
            savingname = path_generic + save_name
        else:
            path_generic = path_res + '/' + 'Generic_function/'
            savingname = path_generic + save_name

        # check whether directory already exist
        if not os.path.exists(path_generic):
            pathlib.Path(path_generic).mkdir(parents=True, exist_ok=True)
        else:
            pass

        # check whether file already exist
        if os.path.isfile(savingname + '.txt') == False:
            pass
        else:
            ls_files_exist = glob(savingname + '*.txt')
            f_exist = [f.split('_')[-1].split('.')[0] for f in ls_files_exist]
            num = 0
            for f in f_exist:
                if 'run' in f:
                    num = int(f.split('run')[-1]) + 1
                else:
                    pass
            savingname = savingname + '_run' + str(num)

        df_out = pd.concat([df_sens1, df_sens2], axis=1, sort=True)
        df_out.to_csv(savingname + '.txt', sep='\t')

        # transform dictionary into array to store it into an hdf5 File
        savingname = savingname + '_report.hdf5'
        f1 = h5py.File(savingname, "w")

        grp_res = f1.create_group("result")
        for m in name_dyes:
            ar_res0 = np.array([dic_fitting['result'][m].index.to_numpy(), dic_fitting['result'][m][0].to_numpy()])
            grp_res.create_dataset(m, ar_res0.shape, data=ar_res0)

        grp_bv = f1.create_group("best values")
        for m in name_dyes:
            subgrp_bv = grp_bv.create_group(m)
            for k, v in dic_fitting['report'][m].best_values.items():
                subgrp_bv.create_dataset(str(k), v.shape, data=np.array(v))
        f1.close()

    return dic_cube, dic_fitting, fig, ax



def generic_function_cube_v2(path_100, pixel_rot, arg, arg_fit, name_dyes, ls_sensor_fit, what_fit, unit, analyte='O2',
                             plot_meas=False, plot_result=False, saving=False, path_res=None):
    """ Radiometrically corrected cubes within the given directory. Then, selection of the highest intensities to
    determine the generic function of the single indicator. he function is assigned to fit 2 single indicators at the
     same time.
    :param path_100:
    :param pixel_90:
    :param arg:
    :param arg_fit:
    :param name_dyes:
    :param what_fit:
    :param path_corr:
    :param averaging:
    :param plot_meas:
    :return:
    """

    # split into regions of  - averaging is required, in particular if multiple RoI for the same indicator are selected
    # use only calibration points that will be used afterwards for averaging
    fit_range = np.arange(arg_fit['fit concentration'][0], arg_fit['fit concentration'][1]+1)

    dic_cube = dict()
    for i in glob(path_100 + '/*.hdr'):
        if np.float(i.split(unit)[0].split('_')[-1]) in fit_range:
            cube, fig, ax = corr.hyperCube_preparation(i, arg=arg, name_dyes=name_dyes, pixel_rot=pixel_rot, save=False,
                                                       averaging=True, plotting=plot_meas, unit=unit, analyte=analyte,
                                                       cube_type='single')

            dic_cube[cube['Concentration']] = cube['average data']

    if len(list(dic_cube.keys())) < 2:
        print('WARNING!')
        print('The standard deviation of the generic function is empty which can cause problems for your evaluation.')

    # ---------------------------------------------------------------------------------
    # relative spectrum to the maximum in the sensor region and include baseline correction before
    # dic_base = dic_cube[k].filter(like=' mean') - dic_cube[k].filter(like=' mean') for k in dic_cube.keys()
    dic_base = dict(map(lambda c: (c, pd.concat([dic_cube[c].filter(like='mean') -
                                                 dic_cube[c].filter(like='mean').min(),
                                                 dic_cube[c].filter(like='STD')], axis=1)), dic_cube.keys()))
    d_norm = [dic_base[k].filter(like=' mean') / dic_base[k].filter(like=' mean').loc[
                                                 arg_fit['fit range sensor'][0]:arg_fit['fit range sensor'][1]].max()
              for k in dic_base.keys()]

    df_sens1 = pd.concat([pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[0]).mean(axis=1),
                          pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[0]).std(axis=1)], axis=1,
                         keys=[ls_sensor_fit[0] + ' mean', ls_sensor_fit[0] + ' STD'])
    df_sens2 = pd.concat([pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[1]).mean(axis=1),
                          pd.concat(d_norm, axis=1).filter(like=ls_sensor_fit[1]).std(axis=1)], axis=1,
                         keys=[ls_sensor_fit[1] + ' mean', ls_sensor_fit[1] + ' STD'])

    # ---------------------------------------------------------------------------------
    # split into regions (ref, sensor, background) for curve fitting
    df_ref, df_rest, df_sens = split_spectrum2parts(df_sens1=df_sens1, df_sens2=df_sens2, ls_sensor_fit=ls_sensor_fit,
                                                    what_fit=what_fit, arg_fit=arg_fit)

    # ---------------------------------------------------------------------------------
    # preparation of what to fit
    what_fit_ = what_fit.copy()
    if 'middle' in what_fit:
        # check whether middle makes sense
        ls_check = []
        for en, x in enumerate(arg_fit['fit range sensor']):
            for em, y in enumerate(arg_fit['fit range ref']):
                ls_check.append(x <= y)

        if all([i == False for i in ls_check]):
            pass
        elif all([i == True for i in ls_check]):
            pass
        else:
            what_fit_.remove('middle')

    # --------------------------------------------------------
    # Curve fitting of averaged generic function
    if len(what_fit_) == 3:
        dic_func = dict(map(lambda d: (d, optimize_3peaks_v2(df_sensor=df_sens[d], df_rest=df_rest[d], df_ref=df_ref[d],
                                                             arg_fit=arg_fit)), ls_sensor_fit))
    elif len(what_fit_) == 2:
        if 'reference' in what_fit:  # no rest / background
            dic_func = dict(map(lambda d: (d, optimize_2peaks_v2(df_sensor=df_sens[d], df_rest=df_ref[d])),
                                ls_sensor_fit))
        else:  # no reference
            dic_func = dict(
                map(lambda d: (d, optimize_2peaks_v2(df_sensor=df_sens[d], df_rest=df_rest[d])), ls_sensor_fit))
    else:
        dic_func = dict(
            map(lambda d: (d, optimize_2peaks_v2(df_sensor=df_sens[d], df_rest=df_rest[d])), ls_sensor_fit))

    # split results and store them into fitting dictionary
    dic_result = pd.concat([dic_func[d][0] for d in ls_sensor_fit], axis=1, keys=ls_sensor_fit)
    dic_result.columns = dic_result.columns.levels[0]
    dic_report = dict(map(lambda d: (d, dic_func[d][1]), ls_sensor_fit))
    df_sens_full = pd.concat([df_sens, df_sens1.filter(like='STD'), df_sens2.filter(like='STD')], axis=1,
                             sort=True).loc[df_sens.index]
    dic_fitting = dict({'result': dic_result, 'report': dic_report, 'sensor': df_sens_full})

    # -----------------------------------------------------------------------
    # Plotting
    if plot_result is True:
        df_std = pd.concat([df_sens1.filter(like='STD'), df_sens2.filter(like='STD')], axis=1,
                           sort=True).loc[arg_fit['fit range sensor'][0]: arg_fit['fit range sensor'][1]]
        df_std.columns = ls_sensor_fit
        df_mean = pd.concat([df_sens1.filter(like='mean'), df_sens2.filter(like='mean')], axis=1,
                            sort=True).loc[arg_fit['fit range sensor'][0]: arg_fit['fit range sensor'][1]]
        df_mean.columns = ls_sensor_fit
        fig, ax = plot.plotting_fit_results_2Sensors(df_sensors=df_mean, df_sensors_std=df_std, arg=arg,
                                                     arg_fit=arg_fit)
    else:
        fig = None
        ax = None

    # -----------------------------------------------------------------------
    # Saving
    if saving is True:
        if path_res is None:
            raise ValueError('Directory to save the output is required. Please provide path_res')
        else:
            save_genericfunction(path_res, dic_cube, df_sens1, df_sens2, dic_fitting)

    return dic_cube, dic_fitting, fig, ax
