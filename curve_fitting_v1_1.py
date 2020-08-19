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
from os import path
import pathlib
import numpy as np
import pandas as pd
import difflib
from lmfit import Model
from scipy.signal import savgol_filter

import math_function_Camera_v1_1 as hycam
import layout_plotting_v1_3 as plot
import correction_hyperCamera_v1_4 as corr


# =====================================================================================
def _tsm_sternvolmer(x, f, m, k):
    """
    fitting function according to the common two site model. In general, x represents the pO2 or pCO2 content, whereas
    m, k and f are the common fitting parameters. the result is given in i/i0 or tau/tau0
    :param x:   list
    :param m:   np.float
    :param k:   np.float
    :param f:   np.float
    :return:
    """
    # i/i0
    return f / (1. + k*x) + (1.-f)/(1. + k*m*x)


def _sternvolmer(x, f, k):
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


def _exponential(x, a, k):
    return a*np.exp(x*k)


# =====================================================================================
# outlier test - Dixon's Q-Test
# source for Q-values http://webspace.ship.edu/pgmarr/Geo441/Tables/Dixon%20Table,%20Expanded.pdf
def q_test_for_smallest_point(dataset, signlevel_pc, num):
    df_qref = pd.read_csv('fitting/Critical_Values_Dixons-QTest.txt', sep='\t', index_col=0, usecols=[0, 1, 2, 3])
    df_qref.columns = [np.float(i) for i in df_qref.columns]

    col = round(1 - signlevel_pc / 100, 2)
    if num in df_qref.index:
        if col in df_qref.columns:
            pass
        else:
            print('Extendet table for q-values is required! Higher significance level needed ', col)
    else:
        print('Extendet table for q-values is required! More samples tested')
    q_ref = df_qref.loc[num, col]
    print('   Confidence level chosen: {:.2f}'.format(q_ref))

    # ----------------------------------------------
    outlier = []
    df = dataset.values
    for i in range(len(df)):
        q_stat = (df[i] - df.min()) / (df.max() - df.min())
        if q_stat > q_ref:
            print('   Outlier detected: remove ', dataset.index[i], ' from dataset')
            outlier.append(dataset.index[i])
    return outlier


# =====================================================================================
# Gaussian line shapes
def _1gaussian_egp(x, amp, cen, wid):
    return amp*(1/(wid*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen)/wid)**2)))


def _1gaussian_v1(x, amp, cen, wid):
    return amp * np.exp(-1*(np.log(2))*((cen-x)*2/wid)**2)


def _2gaussian_v1(x, amp1, cen1, wid1,  amp2, cen2, wid2):
    return amp1 * np.exp(-1*(np.log(2))*((cen1-x)*2/wid1)**2) + amp2 * np.exp(-1*(np.log(2))*((cen2-x)*2/wid2)**2)


# ------------------------------------------------------------
# Lorentzian line shapes
def _1Lorentzian_egp(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)


def _1Lorentzian_v1(x, amp, cen, wid):
    return amp / (1 + ((cen-x)*2/wid)**2)


def _2Lorentzian_v1(x, amp1, cen1, wid1, amp2, cen2, wid2):
    return amp1 / (1 + ((cen1-x)*2/wid1)**2) + amp2 / (1 + ((cen2-x)*2/wid2)**2)


# ------------------------------------------------------------
# Voigt line shapes - combination of gaussian and lorentzian line shapes
def _1Voigt_egp(x, ampG, cenG, sigmaG, ampL, cenL, widL):
    return (ampG*(1/(sigmaG*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG)**2)/((2*sigmaG)**2)))) +\
           (ampL*widL**2/((x-cenL)**2+widL**2))


def _1Voigt_v1(x, weightG, weightL, ampG, cenG, widG, ampL, cenL, widL):
    return weightG*(ampG * np.exp(-1*(np.log(2))*((cenG-x)*2/widG)**2)) +\
           weightL*(ampL / (1 + ((cenL-x)*2/widL)**2))


def _1Voigt_v2(x, weight, ampG, cenG, widG, ampL, cenL, widL):
    return weight*(ampG * np.exp(-1*(np.log(2))*((cenG-x)*2/widG)**2)) +\
           (1-weight)*(ampL / (1 + ((cenL-x)*2/widL)**2))


def _2Voigt_v1(x, weightG1, weightL1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, weightG2, weightL2, ampG2, cenG2, widG2,
               ampL2, cenL2, widL2, weight1, weight2):
    c1 = _1Voigt_v1(x, weightG=weightG1, weightL=weightL1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1,
                    widL=widL1)
    c2 = _1Voigt_v1(x, weightG=weightG2, weightL=weightL2, ampG=ampG2, cenG=cenG2, widG=widG2, ampL=ampL2, cenL=cenL2,
                    widL=widL2)
    return weight1*c1 + weight2*c2


def _2Voigt_v2(x, weight1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, weight2, ampG2, cenG2, widG2, ampL2, cenL2,
               widL2, weight):
    c1 = _1Voigt_v2(x, weight=weight1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1, widL=widL1)
    c2 = _1Voigt_v2(x, weight=weight2, ampG=ampG2, cenG=cenG2, widG=widG2, ampL=ampL2, cenL=cenL2, widL=widL2)
    return weight*c1 + (1-weight)*c2


def _Voigt_gaus_voigt_v1(x, weightG1, weightL1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, amp2, cen2, wid2, weightG3,
                         weightL3, ampG3, cenG3, widG3, ampL3, cenL3, widL3, weight3, weight1, weight2):
    c1 = _1Voigt_v1(x, weightG=weightG1, weightL=weightL1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1,
                    widL=widL1)
    c2 = _1gaussian_v1(x, amp=amp2, cen=cen2, wid=wid2)
    # weightG2, weightL2, ampG2, cenG2, widG2,  ampL2, cenL2, widL2,
    #_1Voigt_v1(x, weightG=weightG2, weightL=weightL2, ampG=ampG2, cenG=cenG2, widG=widG2, ampL=ampL2, cenL=cenL2,
    #                widL=widL2)
    c3 = _1Voigt_v1(x, weightG=weightG3, weightL=weightL3, ampG=ampG3, cenG=cenG3, widG=widG3, ampL=ampL3, cenL=cenL3,
                    widL=widL3)
    return weight1*c1 + weight2*c2 + weight3*c3


def _2Voigt_gauss_v1(x, weightV1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, weightV2, ampG2, cenG2, widG2, ampL2, cenL2,
               widL2, amp3, cen3, wid3, weight1, weight2, weight3):
    c1 = _1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1, widL=widL1)
    c2 = _1Voigt_v2(x, weight=weightV2, ampG=ampG2, cenG=cenG2, widG=widG2, ampL=ampL2, cenL=cenL2, widL=widL2)
    c3 = _1gaussian_v1(x, amp=amp3, cen=cen3, wid=wid3)
    return weight1*c1 + weight2*c2 + weight3*c3


def _Voigt_gauss_v1(x, weightV1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, amp2, cen2, wid2, weight1, weight2):
    c1 = _1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1, widL=widL1)
    c2 = _1gaussian_v1(x, amp=amp2, cen=cen2, wid=wid2)
    return weight1*c1 + weight2*c2


def _2Voigt_lorenz_v1(x, weightG1, weightL1, ampG1, cenG1, widG1, ampL1, cenL1, widL1, weightV, weightL, ampL2, cenL2,
                      widL2):
    c1 = _1Voigt_v1(x, weightG=weightG1, weightL=weightL1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1,
                    widL=widL1)
    c2 = _1Lorentzian_v1(x, amp=ampL2, cen=cenL2, wid=widL2)
    return weightV*c1 + weightL*c2


# -----------------------------------------------------------------
def load_dual_sensor_fit(conc, fit, path, dataset):
    path = path + dataset + '/'
    if dataset == 'individualPixel':
        file_dec = path + conc + '/(Pd-Pt)dual-TPTBP_pixel_' + fit + '-fit.txt'
    else:
        file_dec = path + conc + '/(Pd-Pt)dual-TPTBP_area_' + fit + '-fit.txt'
    print('import', file_dec, '...')
    # --------------------------------------------------------------------
    # general loading of fit results
    df_statistics = pd.DataFrame(np.zeros(shape=(1, 1)), index=['# fitting method'], columns=['deconvolution'])
    en_dec = []
    with open(file_dec, 'r') as file:
        for en, l in enumerate(file):
            if l.startswith('[[Fit Statistics]]'):
                en_dec.append(en)
            if l.startswith('[[Variables]]'):
                en_dec.append(en)
            if l.startswith('[[Correlations]]'):
                en_dec.append(en)
        en_dec.append(en)

    # --------------------------------------------------------------------
    # statistics included
    if len(en_dec) < 4:
        foot1 = en_dec[-1]
    else:
        foot1 = en_dec[-1] - en_dec[1] + en_dec[0]
    l_dec = pd.read_csv(file_dec, skip_blank_lines=True, skiprows=en_dec[0], skipfooter=foot1, engine='python').values

    for i in range(len(l_dec)):
        df_statistics.loc[l_dec[i][0].split('=')[0].strip(), 'deconvolution'] = l_dec[i][0].split('=')[1].strip()

    # --------------------------------------------------------------------
    # parameters
    if len(en_dec) < 4:
        foot = 0
    else:
        foot = en_dec[-1] - en_dec[-2] + 1
    df_params = pd.DataFrame(np.zeros(shape=(1, 1)), columns=['deconvolution'])

    l_dec = pd.read_csv(file_dec, skip_blank_lines=True, skiprows=en_dec[1], skipfooter=foot, engine='python').values
    for i in range(len(l_dec)):
        df_params.loc[l_dec[i][0].strip().split(':')[0].strip(), 'deconvolution'] = np.float(
            l_dec[i][0].strip().split(':')[1].split('+')[0].strip().split(' ')[0])
    df_params = df_params.drop(0)

    # --------------------------------------------------------------------
    # maximal intensity for individual sensors
    x = np.linspace(700, 900, num=int((900 - 700) / 1 + 1))
    col = 'deconvolution'
    if fit == 'lorentz-voigt':
        Pt_dual = _1Lorentzian_v1(x, amp=df_params.loc['ampL2', col], cen=df_params.loc['cenL2', col],
                                  wid=df_params.loc['widL2', col]).max()
        Pd_dual = _1Voigt_v1(x, weightG=df_params.loc['weightG1', col], weightL=df_params.loc['weightL1', col],
                             ampG=df_params.loc['ampG1', col], cenG=df_params.loc['cenG1', col],
                             widG=df_params.loc['widG1', col], ampL=df_params.loc['ampL1', col],
                             cenL=df_params.loc['cenL1', col], widL=df_params.loc['widL1', col]).max()
    elif fit == '2voigt':
        Pt_dual = _1Voigt_v1(x, weightG=df_params.loc['weightG1', col], weightL=df_params.loc['weightL1', col],
                             ampG=df_params.loc['ampG1', col], cenG=df_params.loc['cenG1', col],
                             widG=df_params.loc['widG1', col], ampL=df_params.loc['ampL1', col],
                             cenL=df_params.loc['cenL1', col], widL=df_params.loc['widL1', col]).max()
        Pd_dual = _1Voigt_v1(x, weightG=df_params.loc['weightG2', col], weightL=df_params.loc['weightL2', col],
                             ampG=df_params.loc['ampG2', col], cenG=df_params.loc['cenG2', col],
                             widG=df_params.loc['widG2', col], ampL=df_params.loc['ampL2', col],
                             cenL=df_params.loc['cenL2', col], widL=df_params.loc['widL2', col]).max()
    else:
        raise ValueError('check the fit!')

    return df_params, Pd_dual, Pt_dual


def load_individual_sensors_fit(conc, fit_Pd, fit_Pt, path, dataset):
    path = path + dataset + '/'
    if dataset == 'individualPixel':
        file_pd = path + conc + '/PdTPTBP_pixel_' + fit_Pd + '-fit_700-900nm.txt'
        file_pt = path + conc + '/PtTPTBP_pixel_' + fit_Pt + '-fit_700-900nm.txt'
    else:
        file_pd = path + conc + '/PdTPTBP_area_' + fit_Pd + '-fit_700-900nm.txt'
        file_pt = path + conc + '/PtTPTBP_area_' + fit_Pt + '-fit_700-900nm.txt'
    print('import', file_pd, '...')
    print('import', file_pt, '...')

    # --------------------------------------------------------------------
    # general loading of fit results
    df_statistics = pd.DataFrame(np.zeros(shape=(1, 2)), index=['# fitting method'], columns=['Pd', 'Pt'])
    en_pd = []
    with open(file_pd, 'r') as file:
        for en, l in enumerate(file):
            if l.startswith('[[Fit Statistics]]'):
                en_pd.append(en)
            if l.startswith('[[Variables]]'):
                en_pd.append(en)
            if l.startswith('[[Correlations]]'):
                en_pd.append(en)
        en_pd.append(en)

    en_pt = []
    with open(file_pt, 'r') as file:
        for en, l in enumerate(file):
            if l.startswith('[[Fit Statistics]]'):
                en_pt.append(en)
            if l.startswith('[[Variables]]'):
                en_pt.append(en)
            if l.startswith('[[Correlations]]'):
                en_pt.append(en)
        en_pt.append(en)

    # --------------------------------------------------------------------
    # statistics
    if len(en_pd) < 4:
        foot1 = en_pd[-1]
    else:
        foot1 = en_pd[-1] - en_pd[1]+2

    l_pd = pd.read_csv(file_pd, skip_blank_lines=True, skiprows=en_pd[0], skipfooter=foot1, engine='python').values
    for i in range(len(l_pd)):
        df_statistics.loc[l_pd[i][0].split('=')[0].strip(), 'Pd'] = l_pd[i][0].split('=')[1].strip()

    if len(en_pt) < 4:
        foot2 = en_pt[-1]
    else:
        foot2 = en_pt[-1] - en_pt[1]+2

    l_pt = pd.read_csv(file_pt, skip_blank_lines=True, skiprows=en_pt[0], skipfooter=foot2, engine='python').values
    for i in range(len(l_pt)):
        df_statistics.loc[l_pt[i][0].split('=')[0].strip(), 'Pt'] = l_pt[i][0].split('=')[1].strip()

    # --------------------------------------------------------------------
    # parameters
    if len(en_pd) < 4:
        foot = 0
    else:
        foot = en_pd[-1] - en_pd[-2] + 1
    df_params = pd.DataFrame(np.zeros(shape=(1, 2)), columns=['Pd', 'Pt'])
    l_pd = pd.read_csv(file_pd, skip_blank_lines=True, skiprows=en_pd[1], skipfooter=foot, engine='python').values
    for i in range(len(l_pd)):
        df_params.loc[l_pd[i][0].strip().split(':')[0].strip(), 'Pd'] = np.float(l_pd[i][0].strip().split(':')[1].split('+')[0].strip().split(' ')[0])

    if len(en_pt) < 4:
        foot3 = 0
    else:
        foot3 = en_pt[-1] - en_pt[-2] + 1

    l_pt = pd.read_csv(file_pt, skip_blank_lines=True, skiprows=en_pt[1], skipfooter=foot3, engine='python').values
    for i in range(len(l_pt)):
        df_params.loc[l_pt[i][0].strip().split(':')[0].strip(), 'Pt'] = np.float(l_pt[i][0].strip().split(':')[1].split('+')[0].strip().split(' ')[0])
    df_params = df_params.drop(0)

    # --------------------------------------------------------------------
    # maximal intensity for individual sensors
    x = np.linspace(700, 900, num=int((900 - 700) / 1 + 1))
    sens = 'Pd'
    Pd_inv = _1Voigt_v1(x, weightG=df_params.loc['weightG', sens], weightL=df_params.loc['weightL', sens],
                        ampG=df_params.loc['ampG', sens], cenG=df_params.loc['cenG', sens],
                        widG=df_params.loc['widG', sens],
                        ampL=df_params.loc['ampL', sens], cenL=df_params.loc['cenL', sens],
                        widL=df_params.loc['widL', sens]).max()
    sens = 'Pt'
    if fit_Pt == 'voigt':
        Pt_inv = _1Voigt_v1(x, weightG=df_params.loc['weightG', sens], weightL=df_params.loc['weightL', sens],
                            ampG=df_params.loc['ampG', sens], cenG=df_params.loc['cenG', sens],
                            widG=df_params.loc['widG', sens],
                            ampL=df_params.loc['ampL', sens], cenL=df_params.loc['cenL', sens],
                            widL=df_params.loc['widL', sens]).max()
    elif fit_Pt == 'lorentzian':
        Pt_inv = _1Lorentzian_v1(x, amp=df_params.loc['amp', 'Pt'], cen=df_params.loc['cen', 'Pt'],
                                 wid=df_params.loc['wid', 'Pt']).max()
    return df_params, Pd_inv, Pt_inv


def extract_data_from_dualFit(l_dual, sensorID, ls_conc, what_fit, fit_lambda, fit_conc, baseline_corr=True):
    meas_dual = dict()
    fit_dual = dict()
    meas_ref = dict()
    meas_rest = dict()
    for i in range(len(l_dual)):
        fit_dual[i] = pd.DataFrame(l_dual[i]['fit sensor'][sensorID].best_fit,
                                   index=l_dual[i]['data toFit sensor'][sensorID].index)
        meas_dual[i] = l_dual[i]['data'][sensorID]
        meas_dual[i].columns = [l_dual[i]['concentration']]
        if 'reference' in what_fit:
            meas_ref[i] = pd.DataFrame(l_dual[i]['data toFit ref'][sensorID])
        else:
            meas_ref[i] = None
        if 'middle' in what_fit:
            meas_rest[i] = pd.DataFrame(l_dual[i]['data toFit rest'][sensorID])
        else:
            meas_rest[i] = None

    # -----------------------------------------------
    # selection of fitting range (in terms of concentration)
    # measurement data
    data_rawDual = pd.concat(meas_dual, axis=1)
    data_rawDual.columns = ls_conc
    data_rawDual_0pc = data_rawDual[0].mean(axis=1)
    data_rawDual = data_rawDual.T.drop(0).T
    data_rawDual[0] = data_rawDual_0pc
    data_rawDual = data_rawDual.T.sort_index().T
    data_cropDual = data_rawDual.loc[:, fit_conc[0]:fit_conc[1]]
    data_normDual = data_cropDual / data_cropDual.loc[fit_lambda[0]:fit_lambda[1]].max()

    # curve fitted data
    data_fitDual = pd.concat(fit_dual, axis=1)
    data_fitDual.columns = ls_conc
    data_fitDual_0pc = data_fitDual[0].mean(axis=1)
    data_fitDual = data_fitDual.T.drop(0).T
    data_fitDual[0] = data_rawDual_0pc
    data_fitDual = data_fitDual.T.sort_index().T
    data_fitDual_crop = data_fitDual.loc[:, fit_conc[0]:fit_conc[1]]
    data_fitDual_norm = data_fitDual_crop / data_fitDual_crop.loc[fit_lambda[0]:fit_lambda[1]].max()

    # reference
    data_rawRef = pd.concat(meas_ref, axis=1)
    data_rawRef.columns = ls_conc
    data_rawRef = data_rawRef.T.sort_index().T

    data_rawRef[0] = data_rawRef[0].mean(axis=1)
    data_rawRef = data_rawRef.T.drop_duplicates().T
    data_cropRef = data_rawRef
    data_normRef = data_cropRef / data_cropRef.max()

    # Rest
    data_rawRest = pd.concat(meas_rest, axis=1)
    data_rawRest.columns = ls_conc
    data_rawRest = data_rawRest.T.sort_index().T

    data_rawRest[0] = data_rawRest[0].mean(axis=1)
    data_rawRest = data_rawRest.T.drop_duplicates().T
    data_cropRest = data_rawRest
    data_normRest = data_cropRest / data_cropRest.max()

    if baseline_corr is True:
        print('Baseline correction of measurement data')

        # Dualindicator sensor
        data_rawDual = data_rawDual - data_rawDual.min()
        data_cropDual = data_cropDual - data_cropDual.min()
        data_normDual = data_cropDual / data_cropDual.max()

        # Reference
        data_rawRef = data_rawRef - data_rawRef.min()
        data_cropRef = data_cropRef - data_cropRef.min()
        data_normRef = data_cropRef / data_cropRef.max()

        # Rest
        data_rawRest = data_rawRest - data_rawRest.min()
        data_cropRest = data_cropRest - data_cropRest.min()
        data_normRest = data_cropRest / data_cropRest.max()

    # ---------------------------------------------------
    # combine to dictionary
    dic_dataDual = dict({'rawDual': data_rawDual, 'normDual': data_normDual, 'cropDual': data_cropDual})
    dic_fitDual = dict({'rawDual': data_fitDual, 'normDual': data_fitDual_norm, 'cropDual': data_fitDual_crop})
    dic_dataRef = dict({'rawRef': data_rawRef, 'normRef': data_cropRef, 'cropRef': data_normRef})
    dic_dataRest = dict({'rawRest': data_rawRest, 'normRest': data_cropRest, 'cropRest': data_normRest})

    return dic_dataDual, dic_fitDual, dic_dataRef, dic_dataRest


def save_curve_fitting(arg_fit, conc, itime, save_op, pt_name=None, pd_name=None, dual_name=None, gfit_pt=None,
                       lfit_pt=None, vfit_pt=None, gfit_pd=None, lfit_pd=None, vfit_pd=None, fit_dual=None,
                       fit_comb=None, fig_fit=None, file_hdr=None, eval_strategy=None, save_res=False,
                       save_figure=False):

    if (save_res or save_figure) is True:
        if 'type' in save_op.keys():
            fig_type = save_op['type']
        else:
            fig_type = 'png'
        if 'dpi' in save_op:
            dpi = save_op['dpi']
        else:
            dpi = 300.

        if pt_name is None:
            if pd_name is None:
                if dual_name is None:
                    raise ValueError('At least one sensor ID (single indicator or dual indicator) is required!')

        if eval_strategy is None:
            print('Saving not successful as evaluation strategy is required.')
        else:
            # create folder if it doesn't exist
            if eval_strategy == 'pixel':
                if file_hdr is None:
                    file_path = 'fitting/curvefitting/NewFolder/pixel/' + conc
                else:
                    file_path = 'fitting/curvefitting/' + file_hdr.split('_')[0] + '_' + \
                                file_hdr.split('_')[1].split('/')[0] + '_' + itime + '/pixel/' + conc
            else:
                if file_hdr is None:
                    file_path = 'fitting/curvefitting/NewFolder/area/' + conc
                else:
                    file_path = 'fitting/curvefitting/' + file_hdr.split('_')[0] + '_' +\
                                file_hdr.split('_')[1].split('/')[0] + '_' + itime + '/area/' + conc
            pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

            # define saving name
            if eval_strategy == 'pixel':
                if pt_name is None:
                    if conc.startswith('0'):
                        conc_num = np.float(conc.split('pc')[0]) / 10
                    else:
                        conc_num = np.float(conc.split('pc')[0])

                    fig_name = 'Curvefitting_pixel_' + dual_name + '-' + str(arg_fit['fit range dual'][0]) + '-' + \
                               str(arg_fit['fit range dual'][1]) + '.'
                    fit_name = dual_name + '_pixel_' + str(arg_fit['ls dual'].loc[conc_num].values[0][0]) + '-fit_' + \
                               str(arg_fit['fit range dual'][0]) + '-' + str(arg_fit['fit range dual'][1])
                else:
                    fig_name = 'Curvefitting_pixel_' + pt_name + '-' + str(arg_fit['fit range Pt'][0]) + '-' + \
                           str(arg_fit['fit range Pt'][1]) + '_' + pd_name + '-' + str(arg_fit['fit range Pd'][0]) + \
                           '-' + str(arg_fit['fit range Pd'][1]) + '.'
                    name_ref_pt = pt_name + '_area_fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    gname_pt = pt_name + '_pixel_gaussian-fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    lname_pt = pt_name + '_pixel_lorentzian-fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    vname_pt = pt_name + '_pixel_voigt-fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])

                    name_ref_pd = pd_name + '_pixel_fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    gname_pd = pd_name + '_pixel_gaussian-fit_' + str(arg_fit['fit range Pd'][0]) + '-' + str(
                        arg_fit['fit range Pd'][1])
                    lname_pd = pd_name + '_pixel_lorentzian-fit_' + str(arg_fit['fit range Pd'][0]) + '-' + str(
                        arg_fit['fit range Pd'][1])
                    vname_pd = pd_name + '_pixel_voigt-fit_' + str(arg_fit['fit range Pd'][0]) + '-' + str(
                        arg_fit['fit range Pd'][1])
                    cname = 'Curvefitting_pixel_' + pt_name + '-' + str(arg_fit['fit range Pt'][0]) + '-' + \
                            str(arg_fit['fit range Pt'][1]) + '-' + pd_name + '-' + str(arg_fit['fit range Pd'][0]) +\
                            '-' + str(arg_fit['fit range Pd'][1])
            else:
                if pt_name is None:
                    fig_name = 'Curvefitting_area_' + dual_name + '-' + str(arg_fit['fit range dual'][0]) + '-' + \
                               str(arg_fit['fit range dual'][1]) + '.'
                    if conc.startswith('0'):
                        conc_num = np.float(conc.split('pc')[0]) / 10
                    else:
                        conc_num = np.float(conc.split('pc')[0])

                    fit_name = dual_name + '_area_' + str(arg_fit['ls dual'].loc[conc_num].values[0][0]) + '-fit_' + \
                               str(arg_fit['fit range dual'][0]) + '-' + str(arg_fit['fit range dual'][1])
                else:
                    fig_name = 'Curvefitting_area_' + pt_name + '-' + str(arg_fit['fit range Pt'][0]) + '-' + \
                               str(arg_fit['fit range Pt'][1]) + '_' + pd_name + '-' + str(arg_fit['fit range Pd'][0]) + \
                               '-' + str(arg_fit['fit range Pd'][1]) + '.'
                    name_ref_pt = pt_name + '_area_fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    gname_pt = pt_name + '_area_gaussian-fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    lname_pt = pt_name + '_area_lorentzian-fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    vname_pt = pt_name + '_area_voigt-fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])

                    name_ref_pd = pd_name + '_area_fit_' + str(arg_fit['fit range Pt'][0]) + '-' + str(
                        arg_fit['fit range Pt'][1])
                    gname_pd = pd_name + '_area_gaussian-fit_' + str(arg_fit['fit range Pd'][0]) + '-' + str(
                        arg_fit['fit range Pd'][1])
                    lname_pd = pd_name + '_area_lorentzian-fit_' + str(arg_fit['fit range Pd'][0]) + '-' + str(
                        arg_fit['fit range Pd'][1])
                    vname_pd = pd_name + '_area_voigt-fit_' + str(arg_fit['fit range Pd'][0]) + '-' + str(
                        arg_fit['fit range Pd'][1])
                    cname = 'Curvefitting_area_' + pt_name + '-' + str(arg_fit['fit range Pt'][0]) + '-' + \
                            str(arg_fit['fit range Pt'][1]) + '-' + pd_name + '-' + str(arg_fit['fit range Pd'][0]) +\
                            '-' + str(arg_fit['fit range Pd'][1])

            # save fitting results to folder
            if save_res is True:
                if gfit_pt is None:
                    pass
                else:
                    for k in gfit_pt.keys():
                        if k == pt_name:
                            file_name = file_path + '/' + gname_pt + '_sensor.txt'
                        elif k == 'reference':
                            file_name = file_path + '/' + name_ref_pt + '_reference.txt'
                        else:
                            file_name = file_path + '/' + gname_pt + '_' + k + '.txt'
                        with open(file_name, 'w') as fh:
                            fh.write(gfit_pt[k].fit_report())
                if lfit_pt is None:
                    pass
                else:
                    for k in lfit_pt.keys():
                        if k == pt_name:
                            file_name = file_path + '/' + lname_pt + '_sensor.txt'
                        elif k == 'reference':
                            pass # file_name = file_path + '/' + name_ref_pt + '_reference.txt'
                        else:
                            file_name = file_path + '/' + lname_pt + '_' + k + '.txt'
                        with open(file_name, 'w') as fh:
                            fh.write(lfit_pt[k].fit_report())
                if vfit_pt is None:
                    pass
                else:
                    for k in vfit_pt.keys():
                        if k == pt_name:
                            file_name = file_path + '/' + vname_pt + '_sensor.txt'
                        elif k == 'reference':
                            pass # file_name = file_path + '/' + name_ref_pt + '_reference.txt'
                        else:
                            file_name = file_path + '/' + vname_pt + '_' + k + '.txt'
                        with open(file_name, 'w') as fh:
                            fh.write(vfit_pt[k].fit_report())

                if gfit_pd is None:
                    pass
                else:
                    for k in gfit_pd.keys():
                        if k == pd_name:
                            file_name = file_path + '/' + gname_pd + '_sensor.txt'
                        elif k == 'reference':
                            file_name = file_path + '/' + name_ref_pd + '_reference.txt'
                        else:
                            file_name = file_path + '/' + gname_pd + '_' + k + '.txt'
                        with open(file_name, 'w') as fh:
                            fh.write(gfit_pd[k].fit_report())
                if lfit_pd is None:
                    pass
                else:
                    for k in lfit_pd.keys():
                        if k == pd_name:
                            file_name = file_path + '/' + lname_pd + '_sensor.txt'
                        elif k == 'reference':
                            pass # file_name = file_path + '/' + name_ref_pt + '_reference.txt'
                        else:
                            file_name = file_path + '/' + lname_pd + '_' + k + '.txt'
                        with open(file_name, 'w') as fh:
                            fh.write(lfit_pd[k].fit_report())
                if vfit_pd is None:
                    pass
                else:
                    for k in vfit_pd.keys():
                        if k == pd_name:
                            file_name = file_path + '/' + vname_pd + '_sensor.txt'
                        elif k == 'reference':
                            pass # file_name = file_path + '/' + name_ref_pt + '_reference.txt'
                        else:
                            file_name = file_path + '/' + vname_pd + '_' + k + '.txt'
                        with open(file_name, 'w') as fh:
                            fh.write(vfit_pd[k].fit_report())
                if fit_dual is None:
                    pass
                else:
                    for k in fit_dual.keys():
                        with open(file_path + '/' + fit_name + '_' + k + '.txt', 'w') as fh:
                            fh.write(fit_dual[k].fit_report())

                if fit_comb is None:
                    pass
                else:
                    for k in fit_comb.keys():
                        with open(file_path + '/' + cname + '_' + k.split(' ')[0] + '.txt', 'w') as fh:
                            fh.write(fit_comb[k].fit_report())

            # save figure to folder
            if save_figure is True:
                if fig_fit is None:
                    print('Figure couldn"t be saved. Provide a figure for the saving function!')
                else:
                    if isinstance(fig_type, str):
                        fig_fit.savefig(file_path + '/' + fig_name + fig_type, dpi=dpi)
                    else:
                        for e in fig_type:
                            fig_fit.savefig(file_path + '/' + fig_name + e, dpi=dpi)

    return


# -----------------------------------------------------------------
def _func_object(a, b, df, df_sig1, df_sig2, c):
    sum_ = a * df_sig1 + b * df_sig2
    residual = sum_ - df[c]

    # standard error of regression
    sqr = np.sqrt(sum(np.array([r ** 2 for r in residual.dropna().values])) / (len(residual.dropna()) - 2))
    return sum_, residual, sqr


def _linearcombinationDual(data_super, df_sig1, df_sig2, conc=0, num=100, start_=0., stop_=0.1):
    range_ = np.linspace(start=start_, stop=stop_, num=int(num))
    range_ = list(dict.fromkeys(range_))
    dic_sum_ = dict()
    dic_residual = dict()
    df_sqr = pd.DataFrame(np.zeros(shape=(len(range_), len(range_))), index=range_, columns=range_)

    for a_ in range_:
        for b_ in range_:
            [dic_sum_[(a_, b_)], dic_residual[(a_, b_)],
             df_sqr.loc[a_, b_]] = _func_object(a=a_, b=b_, df_sig1=df_sig1, df_sig2=df_sig2,
                                                df=data_super, c=conc)

    best_lc = df_sqr.min().min()

    for ind_b in df_sqr.columns:
        for ind_a in df_sqr.index:
            if df_sqr.loc[ind_a, ind_b] == best_lc:
                best_a = ind_a
                best_b = ind_b
    dic_params = dict({'sqr': df_sqr, 'bestFit': best_lc, 'Sensor1': best_a, 'Sensor2': best_b})

    return dic_params


def stochastic_parameter_optimization(df_sensor, df_sig1, df_sig2, conc_, nruns, start_0=0., stop_0=1., num=100):

    dic_params_pc = dict()

    for n in range(nruns):
        print('[ Run-', n, ']')
        if n == 0:
            start_n = start_0
            stop_n = stop_0
            dic_params_pc[n] = _linearcombinationDual(data_super=df_sensor, df_sig1=df_sig1, df_sig2=df_sig2, num=num,
                                                      conc=conc_, start_=start_n, stop_=stop_n)
        else:
            step_n = (dic_params_pc[n - 1]['sqr'].index[-1] - dic_params_pc[n - 1]['sqr'].index[0]) / num *10
            start_n = min(dic_params_pc[n - 1]['Sensor1'] - step_n, dic_params_pc[n - 1]['Sensor2'] - step_n)
            stop_n = max(dic_params_pc[n - 1]['Sensor1'] + step_n, dic_params_pc[n - 1]['Sensor2'] + step_n)

            dic_params_pc[n] = _linearcombinationDual(data_super=df_sensor, df_sig1=df_sig1, df_sig2=df_sig2, num=num,
                                                      conc=conc_, start_=start_n, stop_=stop_n)
        # ===========================================================================
        # Result
        print('best Fit: ', dic_params_pc[n]['bestFit'])
        print('Sensor Pt', dic_params_pc[n]['Sensor1'], ' Sensor Pd', dic_params_pc[n]['Sensor2'])

    return dic_params_pc


# --------------------------------------------------------------------------------------------------------------------
def plot_individual_curveFit(data_normPt, data_normPd,arg_fit, arg, what_fit,  df_ref=None, df_rest=None, ls_Pt='voigt',
                             ls_Pd='voigt', fontsize_=13., figsize=(6, 5)):
    fig_fit_Pt = plt.figure(figsize=figsize)
    axPt = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    axPt_res = plt.subplot2grid((4, 4), (2, 0), colspan=2)

    print('Curve fitting of Pt-sensor')
    [dic_result_pt, dic_tofit_pt,
     para_pt] = _curvefit_sensor(df_sensor=data_normPt, arg_fit=arg_fit, arg=arg, ax=axPt, df_ref=df_ref,
                                 df_rest=df_rest, col_data=arg['color Pt'], lineshape=ls_Pt, plot_fig=True,
                                 fitting_range=arg_fit['fit range Pt'], ax_res=axPt_res,
                                 fig=fig_fit_Pt, what_fit=what_fit)
    axPt.set_title('X2 = {:.2e}'.format(dic_result_pt['result sensor'].redchi), loc='left', fontsize=fontsize_ * .8)
    axPt.get_legend().remove()
    plt.tight_layout(h_pad=-2.2)
    plt.show()

    # ----------------------------------------------------------------
    fig_fit_Pd = plt.figure(figsize=figsize)
    axPd = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    axPd_res = plt.subplot2grid((4, 4), (2, 0), colspan=2)

    print('Curve fitting of Pd-sensor')
    [dic_result_pd, dic_tofit_pd,
     para_pd] = _curvefit_sensor(df_sensor=data_normPd, arg_fit=arg_fit, arg=arg, what_fit=what_fit, df_ref=df_ref,
                                 df_rest=df_rest, col_data=arg['color Pd'], lineshape=ls_Pd, plot_fig=True, ax=axPd,
                                 fitting_range=arg_fit['fit range Pd'], ax_res=axPd_res,
                                 fig=fig_fit_Pd)

    axPd.set_title('X2 = {:.2e}'.format(dic_result_pd['result sensor'].redchi), loc='left',
                   fontsize=fontsize_ * .8)
    axPd.get_legend().remove()
    plt.tight_layout(h_pad=-2.2)
    plt.show()

    return fig_fit_Pt, fig_fit_Pd, dic_result_pt, dic_result_pd


# ---------------------------------------------------------------------------------------------------------------------
def _curvefit_sensor(df_sensor, arg_fit, arg, what_fit, col_data='darkorange', lineshape='voigt', df_paraD=None,
                     df_ref=None, df_rest=None, plot_fig=False, ax=None, fitting_range=(700, 900), ax_res=None,
                     fig=None, plot_data_ref=False):
    # dictionary to store them all
    dic_toFit = dict()
    dic_result = dict()

    if 'reference' in what_fit:
        model = Model(_1Voigt_v1)
        ampG = df_ref.max().values[0]
        cenG = df_ref.idxmax().values[0]
        widG = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2
        ampL = df_ref.loc[arg_fit['fit range ref'][0] + widG:arg_fit['fit range ref'][1]].max().values[0]
        cenL = df_ref.loc[arg_fit['fit range ref'][0] + widG:arg_fit['fit range ref'][1]].idxmax().values[0]
        widL = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2
        params = model.make_params(weightG=0.5, weightL=0.5, ampG=ampG, cenG=cenG, widG=widG, ampL=ampL, cenL=cenL,
                                   widL=widL)
        # --------------------------------------------------------
        # lmfit - fitting
        df_tofit_ref = df_ref[df_ref.columns[0]]
        result_ref = model.fit(df_tofit_ref.values.tolist(), params, x=np.array(df_tofit_ref.index), nan_policy='omit')

        dic_toFit['toFit ref'] = df_tofit_ref
        dic_result['result ref'] = result_ref

    if 'middle' in what_fit:
        model = Model(_1Voigt_v1)

        widG = (df_rest.index[-1] - df_rest.index[0]) / 2
        widL = (df_rest.index[-1] - df_rest.index[0]) / 2
        ampG = df_rest.loc[:widG + df_rest.index[0]].max().values[0]
        cenG = df_rest.loc[:widG + df_rest.index[0]].idxmax().values[0]

        ampL = df_rest.loc[widG + df_rest.index[0] + 20:].max().values[0]
        cenL = df_rest.loc[widG + df_rest.index[0] + 20:].idxmax().values[0]
        params_mid = model.make_params(weightG=0.5, weightL=0.5, ampG=ampG, cenG=cenG, widG=widG, ampL=ampL, cenL=cenL,
                                       widL=widL)
        # lmfit - fitting
        df_tofit_rest = df_rest[df_rest.columns[0]]
        result_rest = model.fit(df_tofit_rest.values.tolist(), params_mid, x=np.array(df_tofit_rest.index),
                                nan_policy='omit')

        dic_toFit['toFit middle'] = df_tofit_rest
        dic_result['result middle'] = result_rest

    if 'dual' in what_fit or 'single' in what_fit:
        # model for lmfit
        if lineshape == 'gaussian':
            model = Model(_1gaussian_v1)
        elif lineshape == 'lorentzian':
            model = Model(_1Lorentzian_v1)
        elif lineshape == 'voigt':
            model = Model(_1Voigt_v1)
        elif lineshape == 'gaussian-gaussian':
            model = Model(_2gaussian_v1)
        elif lineshape == 'lorentzian-lorentzian':
            model = Model(_2Lorentzian_v1)
        elif lineshape == '1voigt':
            model = Model(_2Voigt_v1)
        elif lineshape == 'voigt-voigt':
            model = Model(_2Voigt_v1)
        elif lineshape == 'voigt-lorentzian' or lineshape == 'lorentzian-voigt':
            model = Model(_2Voigt_lorenz_v1)
        else:
            raise ValueError('choose correct model for curve fitting: gaussian, lorentzian or voigt. For dualFit choose'
                             'either 2voigt or a combination such as voigt-lorentz or lorentz-voigt')

        # --------------------------------------------------------------
        # lmfit - parameter setting
        if lineshape == 'voigt':
            widG = (df_sensor.index[-1] - df_sensor.index[0]) / 2
            widL = (df_sensor.index[-1] - df_sensor.index[0]) / 2

            if 'ampG' in arg_fit.keys():
                ampG = arg_fit['ampG']
            else:
                ampG = df_sensor.loc[:widG + df_sensor.index[0]].max()
            if 'ampL' in arg_fit.keys():
                ampL = arg_fit['ampL']
            else:
                ampL = df_sensor.loc[widG + df_sensor.index[0] + 20:].max()
            if 'cenG' in arg_fit.keys():
                cenG = arg_fit['cenG']
            else:
                cenG = df_sensor.loc[:widG + df_sensor.index[0]].idxmax()

            if 'cenL' in arg_fit.keys():
                cenL = arg_fit['cenL']
            else:
                cenL = df_sensor.loc[widG + df_sensor.index[0] + 20:].idxmax()

            params = model.make_params(weightG=0.5, weightL=0.5, ampG=ampG, cenG=cenG, widG=widG, ampL=ampL, cenL=cenL,
                                       widL=widL)
            params['ampG'].max = 1.
            params['ampG'].min = 0.
            params['ampL'].max = 1.
            params['ampL'].min = 0.

        elif lineshape == 'gaussian':
            if 'ampG' in arg_fit.keys():
                amp = arg_fit['ampG']
            else:
                raise ValueError('amp is missing')
            if 'cenG' in arg_fit.keys():
                cen = arg_fit['cenG']
            else:
                raise ValueError('cen is missing')
            if 'widG' in arg_fit.keys():
                wid = arg_fit['widG']
            else:
                raise ValueError('wid is missing')
            params = model.make_params(amp=amp, cen=cen, wid=wid)
            params['amp'].max = 1.
            params['amp'].min = 0.

        elif lineshape == 'lorentzian':
            if 'ampL' in arg_fit.keys():
                amp = arg_fit['ampL']
            else:
                raise ValueError('amp is missing')
            if 'cenL' in arg_fit.keys():
                cen = arg_fit['cenL']
            else:
                raise ValueError('cen is missing')
            if 'widL' in arg_fit.keys():
                wid = arg_fit['widL']
            else:
                raise ValueError('wid is missing')
            params = model.make_params(amp=amp, cen=cen, wid=wid)
            params['amp'].max = 1.
            params['amp'].min = 0.

        elif lineshape == 'voigt-voigt':
            # lower wavelength dominated by Pt
            weightG1 = df_paraD.loc['weightG', 'Pt']
            weightL1 = df_paraD.loc['weightL', 'Pt']
            ampG1 = df_paraD.loc['ampG', 'Pt']
            ampL1 = df_paraD.loc['ampL', 'Pt']
            cenG1 = df_paraD.loc['cenG', 'Pt']
            cenL1 = df_paraD.loc['cenL', 'Pt']
            widG1 = df_paraD.loc['widG', 'Pt']
            widL1 = df_paraD.loc['widL', 'Pt']
            # higher wavelength dominated by Pd
            weightG2 = df_paraD.loc['weightG', 'Pd']
            weightL2 = df_paraD.loc['weightL', 'Pd']
            ampG2 = df_paraD.loc['ampG', 'Pd']
            ampL2 = df_paraD.loc['ampL', 'Pd']
            cenG2 = df_paraD.loc['cenG', 'Pd']
            cenL2 = df_paraD.loc['cenL', 'Pd']
            widG2 = df_paraD.loc['widG', 'Pd']
            widL2 = df_paraD.loc['widL', 'Pd']

            params = model.make_params(weightG1=weightG1, weightL1=weightL1, ampG1=ampG1,  cenG1=cenG1, widG1=widG1,
                                       ampL1=ampL1, cenL1=cenL1, widL1=widL1, weightG2=weightG2, weightL2=weightL2,
                                       ampG2=ampG2, cenG2=cenG2, widG2=widG2, ampL2=ampL2, cenL2=cenL2, widL2=widL2,
                                       weight1=1, weight2=1)
            params['ampG1'].max = 1.
            params['ampG2'].max = 1.
            params['ampL1'].max = 1.
            params['ampL2'].max = 1.
            params['ampG1'].min = 0.
            params['ampG2'].min = 0.
            params['ampL1'].min = 0.
            params['ampL2'].min = 0.

        elif lineshape == 'voigt-lorentzian':
            weightPt = 1.
            weightPd = 1.
            # lower wavelength dominated by Pt
            weightG1 = df_paraD.loc['weightG', 'Pt']
            weightL1 = df_paraD.loc['weightL', 'Pt']
            ampG1 = df_paraD.loc['ampG', 'Pt']
            ampL1 = df_paraD.loc['ampL', 'Pt']
            cenG1 = df_paraD.loc['cenG', 'Pt']
            cenL1 = df_paraD.loc['cenL', 'Pt']
            widG1 = df_paraD.loc['widG', 'Pt']
            widL1 = df_paraD.loc['widL', 'Pt']
            # higher wavelength dominated by Pd
            ampL2 = df_paraD.loc['amp', 'Pd']
            cenL2 = df_paraD.loc['cen', 'Pd']
            widL2 = df_paraD.loc['wid', 'Pd']

            params = model.make_params(weightG1=weightG1, weightL1=weightL1, ampG1=ampG1, cenG1=cenG1, widG1=widG1,
                                             ampL1=ampL1, cenL1=cenL1, widL1=widL1, weightV=weightPt, weightL=weightPd,
                                             ampL2=ampL2, cenL2=cenL2, widL2=widL2)

        elif lineshape == 'lorentzian-voigt':
            weightPt = 1.
            weightPd = 1.
            # lower wavelength dominated by Pt
            ampL2 = df_paraD.loc['amp', 'Pt']
            cenL2 = df_paraD.loc['cen', 'Pt']
            widL2 = df_paraD.loc['wid', 'Pt']
            # higher wavelength dominated by Pd
            weightG1 = df_paraD.loc['weightG', 'Pd']
            weightL1 = df_paraD.loc['weightL', 'Pd']
            ampG1 = df_paraD.loc['ampG', 'Pd']
            ampL1 = df_paraD.loc['ampL', 'Pd']
            cenG1 = df_paraD.loc['cenG', 'Pd']
            cenL1 = df_paraD.loc['cenL', 'Pd']
            widG1 = df_paraD.loc['widG', 'Pd']
            widL1 = df_paraD.loc['widL', 'Pd']

            params = model.make_params(weightG1=weightG1, weightL1=weightL1, ampG1=ampG1, cenG1=cenG1, widG1=widG1,
                                       ampL1=ampL1, cenL1=cenL1, widL1=widL1, weightV=weightPt, weightL=weightPd,
                                       ampL2=ampL2, cenL2=cenL2, widL2=widL2)

        else:
            print('work in progress...')
            params = model.make_params(cen=df_sensor.idxmax().values[0], amp=df_sensor.max().values[0], wid=30.)

        # --------------------------------------------------------------
        # lmfit - fitting
        df_tofit_sensor = df_sensor.loc[fitting_range[0]:fitting_range[1]]
        result_sensor = model.fit(df_tofit_sensor.values.tolist(), params, x=np.array(df_tofit_sensor.index),
                                  nan_policy='omit')
        dic_toFit['toFit sensor'] = df_tofit_sensor
        dic_result['result sensor'] = result_sensor

    # --------------------------------------------------------------
    # plotting results
    if plot_fig is True:
        fig, ax, ax_dev = plot.plotting_fitresults(xdata=df_tofit_sensor.index, ddf=df_sensor, result=result_sensor,
                                                   fig=fig, arg=arg, fit=lineshape, ax=ax, ax_dev=ax_res,
                                                   col_data=col_data)
        if 'reference' in what_fit:
            fig, ax, ax_dev = plot.plotting_fitresults(xdata=df_tofit_ref.index, ddf=df_sensor, result=result_ref,
                                                       fig=fig, arg=arg, fit='reference', ax=ax, ax_dev=ax_res,
                                                       col_data=col_data, plot_data=plot_data_ref)
        if 'middle' in what_fit:
            fig, ax, ax_dev = plot.plotting_fitresults(xdata=df_tofit_rest.index, ddf=df_sensor, result=result_rest,
                                                       fig=fig, arg=arg, fit='middle', ax=ax, ax_dev=ax_res,
                                                       col_data='slategrey', plot_data=False)
        plot_para = dict({'figure': fig, 'ax': ax, 'residuals': ax_dev})
    else:
        plt.ioff()
        fig, ax, ax_dev = plot.plotting_fitresults(xdata=df_tofit_sensor.index, ddf=df_sensor, result=result_sensor,
                                                   fig=fig, arg=arg, fit=lineshape, ax=ax, ax_dev=ax_res,
                                                   col_data=col_data)
        if 'reference' in what_fit:
            fig, ax, ax_dev = plot.plotting_fitresults(xdata=df_tofit_ref.index, ddf=df_sensor, result=result_ref,
                                                       fig=fig, arg=arg, fit='reference', ax=ax, ax_dev=ax_res,
                                                       color_fit=col_data, plot_data=plot_data_ref)
        if 'middle' in what_fit:
            fig, ax, ax_dev = plot.plotting_fitresults(xdata=df_tofit_rest.index, ddf=df_sensor, result=result_rest,
                                                       fig=fig, arg=arg, fit='middle', ax=ax, ax_dev=ax_res,
                                                       color_fit='slategrey', plot_data=False)
        plot_para = dict({'figure': fig, 'ax': ax, 'residuals': ax_dev})
        plt.close(fig)

    return dic_result, dic_toFit, plot_para


def curve_fitting_indvidual(file_hdr, name_dyes, arg, arg_fit, what_fit, corr_file, eval_strategy='pixel',
                            plot_cube=False, pixel=None, plot_measurement=False, plot_res=False, save_op=None,
                            save_res=False, save_figure=False, corr_type='extrapolated'):
    print('Analyzing: ', file_hdr)
    what_fit_ = what_fit.copy()
    # --------------------------------------------------------------
    # preparation
    # pre-check parameter constellation
    if (plot_cube or plot_measurement) is True:
        if pixel is None:
            pixel = [[(200, 696), (150, 850), (230, 750)], [(400, 821), (450, 710), (500, 700)],
                     [(800, 778), (760, 700), (850, 870)]]
            # raise ValueError('Define pixel for evaluation')
        if 'ls' not in arg.keys():
            raise ValueError('Define line style for evaluation')
    if pixel is None:
        pixel = [[(200, 696), (150, 850), (230, 750)], [(400, 821), (450, 710), (500, 700)],
                 [(800, 778), (760, 700), (850, 870)]]

    # define required parameter
    conc = file_hdr.split('Sensor_')[1].split('_')[1] # file_hdr.split('/')[2].split('_')[2]

    # --------------------------------------------------------------
    # load cube
    para = corr.load_cube(file_hdr=file_hdr, rotation=90., corr_file=corr_file, plot_cube=plot_cube)
    itime = str(int(para['Integration time'])) + 'ms'

    # correction of measurement signal and extracting individual pixels
    plt.ioff()
    [df_pixel, df_pixel_corr, fig_uncorr,
     fig_corr] = hycam.calibration_solution(para=para, pixel=pixel, name_dyes=name_dyes, ls=arg['ls'], arg=arg,
                                            colors=arg['colors'], plotting='both', plot_cube=plot_cube)
    plt.close(fig_corr)
    plt.close(fig_uncorr)

    # =====================================================================================================
    # how to treat the data
    # either smooth individual pixel (savgol_filter + averaging) or load defined areas (done in HSI Studio)
    # --------------------------------------------------------------
    # individual pixel
    id_pt = pd.DataFrame(np.zeros(shape=(0, 0)))
    id_pd = pd.DataFrame(np.zeros(shape=(0, 0)))

    if eval_strategy == 'pixel':
        # Pt-indicator
        check = 'Pt'
        for en, idx in enumerate(name_dyes):
            if idx[:2].lower() == check.lower():
                id_pt.loc[idx, 0] = int(en)

        dev_name = dict()
        string_dev = dict()
        dict_pt = dict()
        for en, tag in enumerate(id_pt.index):
            dict_pt[tag] = None

        for i in range(len(id_pt.index)):
            dev_name[i] = [li for li in difflib.ndiff(id_pt.index[i], min(id_pt.index, key=len)) if li[0] != ' ']
            if '- +' in dev_name:
                dev_name[i].remove('- +')
            string_dev[i] = ''.join([i.split(' ')[1] for i in dev_name[i]])

        for en in string_dev:
            sensor_id = [x for x in id_pt.index if string_dev[en] in x]
            if len(sensor_id) == 1:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_pt.loc[sensor_id, 0].values[0])]].mean(axis=1), 7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_pt[sensor_id[0]] = df_
            else:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_pt.loc[min(id_pt.index, key=len), 0])]].mean(axis=1), 7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_pt[min(id_pt.index, key=len)] = df_

        df_Pt = pd.concat(dict_pt, axis=1, sort=False)
        df_Pt.columns = [col[0] for col in df_Pt.columns]

        # ---------------------------
        # Pd-indicator
        check = 'Pd'
        for en, idx in enumerate(name_dyes):
            if idx[:2].lower() == check.lower():
                id_pd.loc[idx, 0] = int(en)

        dev_name = dict()
        string_dev = dict()
        dict_pd = dict()
        for en, tag in enumerate(id_pd.index):
            dict_pd[tag] = None

        for i in range(len(id_pd.index)):
            dev_name[i] = [li for li in difflib.ndiff(id_pd.index[i], min(id_pd.index, key=len)) if li[0] != ' ']
            if '- +' in dev_name:
                dev_name[i].remove('- +')
            string_dev[i] = ''.join([i.split(' ')[1] for i in dev_name[i]])

        for en in string_dev:
            sensor_id = [x for x in id_pd.index if string_dev[en] in x]

            if len(sensor_id) == 1:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_pd.loc[sensor_id, 0].values[0])]].mean(axis=1), 7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_pd[sensor_id[0]] = df_
            else:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_pd.loc[min(id_pd.index, key=len), 0])]].mean(axis=1), 7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_pd[min(id_pd.index, key=len)] = df_

        df_Pd = pd.concat(dict_pd, axis=1, sort=False)
        df_Pd.columns = [col[0] for col in df_Pd.columns]

        # ---------------------------
        # Pt/Pd-dual-indicator
        name_tag = name_dyes.copy()
        for p in id_pt.index.tolist() + id_pd.index.tolist():
            name_tag.remove(p)

        id_dual = pd.DataFrame(np.zeros(shape=(len(name_tag), 0)), index=name_tag)
        check = name_tag
        for c in range(len(check)):
            for en, idx in enumerate(name_dyes):
                if idx.lower() == check[c].lower():
                    id_dual.loc[idx, 0] = int(en)

        dev_name = dict()
        string_dev = dict()
        dict_dual = dict()
        for en, tag in enumerate(id_dual.index):
            dict_dual[tag] = None

        for i in range(len(id_dual.index)):
            dev_name[i] = [li for li in difflib.ndiff(id_dual.index[i], min(id_dual.index, key=len)) if li[0] != ' ']
            if '- +' in dev_name:
                dev_name[i].remove('- +')
            string_dev[i] = ''.join([i.split(' ')[1] for i in dev_name[i]])

        for en in string_dev:
            sensor_id = [x for x in id_dual.index if string_dev[en] in x]

            if len(sensor_id) == 1:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_dual.loc[sensor_id, 0].values[0])]].mean(axis=1), 7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_dual[sensor_id[0]] = df_
            else:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_dual.loc[min(id_dual.index, key=len), 0])]].mean(axis=1),
                                   7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_dual[min(id_dual.index, key=len)] = df_

        df_dual = pd.concat(dict_dual, axis=1, sort=False)
        df_dual.columns = [col[0] for col in df_dual.columns]

    # --------------------------------------------------------------
    elif eval_strategy == 'area':
        file_area = '_'.join(file_hdr.split('cube')[0].split('pc')[0].split('_')[:-1]) + '_' + conc + '_spectrum.csv'
        if path.exists(file_area) is True:
            pass
        else:
            file_area = '_'.join(file_hdr.split('cube')[0].split('pc')[0].split('_')[:-1]) + '_' + conc + '_spectra.csv'
            if path.exists(file_area) is True:
                pass
            else:
                raise ValueError('Check path+filename of spectra')

        df_ = pd.read_csv(file_area, header=None).T.loc[4:].set_index(0)
        col_name = pd.read_csv(file_area, header=None).T.loc[0].values[1:]
        df_.columns = col_name

        # correction of regions
        kappa = para['correction']['correction factor']
        index_new = [round(i, 3) for i in kappa.index]
        kappa.index = index_new
        df_corr = hycam.cube_signal_correction(df_uncorr=df_, kappa=kappa)
        df_corr.index.name = None
        df_corr = df_corr.astype(float)

        # Pt-indicator
        check = 'Pt'
        id_pt = pd.DataFrame(np.zeros(shape=(0, 0)))
        for en, idx in enumerate(col_name):
            if idx[:2].lower() == check.lower():
                id_pt.loc[idx, 0] = int(en)

        # Pd-indicator
        check = 'Pd'
        id_pd = pd.DataFrame(np.zeros(shape=(0, 0)))
        for en, idx in enumerate(col_name):
            if idx[:2].lower() == check.lower():
                id_pd.loc[idx, 0] = int(en)

        # Pt/Pd-Dualindicator
        check = 'Du'
        name_tag = name_dyes.copy()
        for p in id_pt.index.tolist() + id_pd.index.tolist():
            name_tag.remove(p)
        id_dual = pd.DataFrame(np.zeros(shape=(len(name_tag), 0)), index=name_tag)
        for en, idx in enumerate(col_name):
            if idx[:2].lower() == check.lower():
                id_dual.loc[idx, 0] = int(en)

        df_Pt = pd.DataFrame(df_corr[id_pt.index.tolist()])
        df_Pd = pd.DataFrame(df_corr[id_pd.index.tolist()])
        df_dual = pd.DataFrame(df_corr[id_dual.index.tolist()])

        dev_name = dict()
        string_dev = []
        for i in range(len(id_pt.index)):
            dev_name[i] = [li for li in difflib.ndiff(id_pt.index[i], min(id_pt.index, key=len)) if li[0] != ' ']
            if '- +' in dev_name:
                dev_name[i].remove('- +')
            string_dev.append(''.join([i.split(' ')[1] for i in dev_name[i]]))

    # --------------------------------------------------------------
    else:
        raise ValueError('Define evaluation strategy, whether individual pixel or a whole (pre-defined) region should '
                         'be analyzed: eval_strategy = pixel /area')

    # --------------------------------------------------------------
    # Plot measurement
    if plot_measurement is True:
        for en, tag in enumerate(df_Pt):
            arg['label'] = [tag, df_Pd.columns[en], df_dual.columns[en]]
            fig, ax = plot.plotting_3optodes(arg=arg, df_sensor1=df_Pt[tag], df_sensor2=df_Pd[df_Pd.columns[en]],
                                             df_sensor3=df_dual[df_dual.columns[en]])
    else:
        for en, tag in enumerate(df_Pt):
            arg['label'] = [tag, df_Pd.columns[en], df_dual.columns[en]]
            plt.ioff()
            fig, ax = plot.plotting_3optodes(arg=arg, df_sensor1=df_Pt[tag], df_sensor2=df_Pd[df_Pd.columns[en]],
                                             df_sensor3=df_dual[df_dual.columns[en]])
            plt.close(fig)

    # =================================================================================================================
    # combining to dictionary
    dic_fitting = dict({'concentration': conc, 'cube': para, 'Pt data': df_Pt, 'Pd data': df_Pd})

    # =================================================================================================================
    # individual parameter fit - for each set
    dic_bestfit_all = dict()

    for en, c in enumerate(df_Pt.columns):
        # ---------------------------------------------------------
        # splitting function into parts
        if max(string_dev, key=len) in c:
            if 'reference' in what_fit_:
                pass
            else:
                what_fit_.append('reference')
            df_ref = pd.DataFrame(df_dual[df_dual.columns[en]].loc[arg_fit['fit range ref'][0]:arg_fit['fit range ref'][1]].astype(float))
        else:
            if 'reference' in what_fit_:
                what_fit_.remove('reference')
            df_ref = None
        if 'middle' in what_fit_:
            df_rest = pd.DataFrame(df_dual[df_dual.columns[en]].loc[arg_fit['fit range ref'][1]:arg_fit['fit range dual'][0]].astype(float))
        else:
            df_rest = None

        df_senPt = df_Pt[df_Pt.columns[en]].loc[df_rest.index[0]:].loc[arg_fit['fit range Pt'][0]:arg_fit['fit range Pt'][1]]
        df_senPd = df_Pd[df_Pd.columns[en]].loc[df_rest.index[0]:].loc[arg_fit['fit range Pd'][0]:arg_fit['fit range Pd'][1]]

        if df_ref is None:
            df_resultPt, result_repPt, df_tofitPt = optimize_2peaks(df_sensor=df_senPt, df_rest=df_rest)
            df_resultPd, result_repPd, df_tofitPd = optimize_2peaks(df_sensor=df_senPd, df_rest=df_rest)
        else:
            df_resultPt, result_repPt, df_tofitPt = optimize_3peaks(df_sensor=df_senPt, df_rest=df_rest, df_ref=df_ref,
                                                                    arg_fit=arg_fit)
            df_resultPd, result_repPd, df_tofitPd = optimize_3peaks(df_sensor=df_senPd, df_rest=df_rest, df_ref=df_ref,
                                                                    arg_fit=arg_fit)

        df_result_all = pd.concat([df_resultPt, df_resultPd], axis=1)
        df_result_all.columns = [c, 'Pd-'+c.split('-')[1]]

        # --------------------------------------------------------------
        # Plotting results
        if plot_res is True:
            pass
        else:
            plt.ioff()
        fig_fit = plt.figure(figsize=arg['figure size Fit'])
        ax0 = plt.subplot2grid((11, 5), (0, 0), colspan=2, rowspan=2)
        ax0_res = plt.subplot2grid((11, 5), (2, 0), colspan=2)
        ax1 = plt.subplot2grid((11, 5), (4, 0), colspan=2, rowspan=2)
        ax1_res = plt.subplot2grid((11, 5), (6, 0), colspan=2)

        if df_ref is None:
            pass
        else:
            ax0.plot(df_ref, marker='.', fillstyle='none', color='navy', lw=0, label='reference')
            ax1.plot(df_ref, marker='.', fillstyle='none', color='navy', lw=0, label='reference')
        ax0.plot(df_rest, marker='o', fillstyle='none', color='grey', lw=0, label='background')
        ax1.plot(df_rest, marker='o', fillstyle='none', color='grey', lw=0, label='background')

        # sensor1: Pt-TPTBP
        ax0.set_title(c, loc='left', fontsize=arg['fontsize Fit'] * 1.1)
        fig_fit, ax0, ax0_res = plot.plotting_fitresults(xdata=df_resultPt.index, ddf=df_senPt, result=result_repPt,
                                                         fig=fig_fit, arg=arg, fit='Voigt', ax=ax0, ax_dev=ax0_res,
                                                         col_data=arg['color Pt'])

        # sensor2: Pd-TPTBP
        ax1.set_title('Pd-'+c.split('-')[1], loc='left', fontsize=arg['fontsize Fit'] * 1.1)
        fig_fit, ax1, ax1_res = plot.plotting_fitresults(xdata=df_resultPd.index, ddf=df_senPd, result=result_repPd,
                                                         fig=fig_fit, arg=arg, fit='Voigt', ax=ax1, ax_dev=ax1_res,
                                                         col_data=arg['color Pd'])

        ax0.legend(loc=0, frameon=True, fancybox=True)
        ax1.legend(loc=0, frameon=True, fancybox=True)
        plt.tight_layout(w_pad=-5.)
        if plot_res is True:
            plt.show()
        else:
            plt.close(fig_fit)

        # =============================================================================================================
        # combining to dictionary
        if 'middle' in what_fit_:
            dic_fitting['background data'] = df_rest
        if 'reference' in what_fit_:
            dic_fitting['reference data'] = df_ref

        dic_bestfit_all[c] = df_result_all
        if len(c.split('+')) == 1:
            l_ = dict({c + ' report': result_repPt, 'Pd-' + c.split('-')[1] + ' report': result_repPd})
            dic_fitting['report without ref'] = l_
        else:
            l_ = dict({c + ' report': result_repPt, 'Pd-' + c.split('-')[1] + ' report': result_repPd})
            dic_fitting['report ' + c.split('+')[1]] = l_

        # saving
        save_curve_fitting(pt_name=c, pd_name='Pd-'+c.split('-')[1], arg_fit=arg_fit, conc=conc, itime=itime,
                           save_op=save_op, fig_fit=fig_fit, file_hdr=file_hdr, save_res=save_res, fit_comb=l_,
                           eval_strategy=eval_strategy, save_figure=save_figure)

    return dic_fitting, dic_bestfit_all


def optimize_2peaks(df_sensor, df_rest):
    # noise + Sensor
    model2 = Model(_Voigt_gauss_v1)

    # sensor - voigt Fit
    weightV1 = 0.5
    widG1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widL1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    ampG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].max()
    cenG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].idxmax()
    ampL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].max()
    cenL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].idxmax()

    # noise - gauss Fit
    wid2 = (df_rest.index[-1] - df_rest.index[0])
    amp2 = df_rest.loc[:wid2 + df_rest.index[0]].max().values[0]
    cen2 = df_rest.loc[:wid2 + df_rest.index[0]].idxmax().values[0]

    params2 = model2.make_params(weightV1=weightV1, ampG1=ampG1, cenG1=cenG1, widG1=widG1, ampL1=ampL1, cenL1=cenL1,
                                 widL1=widL1, amp2=amp2, cen2=cen2, wid2=wid2, weight1=1., weight2=0.5)

    # define the frame of the sensor curves
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
    df_tofit = pd.concat([df_sensor, df_rest], axis=0).mean(axis=1)
    df_tofit = df_tofit.sort_index()
    result_ = model2.fit(df_tofit.values.tolist(), params2, x=np.array(df_tofit.index), nan_policy='omit')
    df_result = pd.DataFrame(result_.best_fit, index=df_tofit.index)

    return df_result, result_, df_tofit


def optimize_2peaks_v2(df_sensor, df_rest):
    # noise + Sensor
    model2 = Model(_Voigt_gauss_v1)

    # sensor - voigt Fit
    weightV1 = 0.5
    widG1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widL1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    ampG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].max()
    cenG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].idxmax()
    ampL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].max()
    cenL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].idxmax()

    # noise - gauss Fit
    wid2 = (df_rest.index[-1] - df_rest.index[0])
    amp2 = df_rest.loc[:wid2 + df_rest.index[0]].max()
    cen2 = df_rest.loc[:wid2 + df_rest.index[0]].idxmax()

    params2 = model2.make_params(weightV1=weightV1, ampG1=ampG1, cenG1=cenG1, widG1=widG1, ampL1=ampL1, cenL1=cenL1,
                                 widL1=widL1, amp2=amp2, cen2=cen2, wid2=wid2, weight1=1., weight2=0.5)

    # define the frame of the sensor curves
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


def optimize_3peaks(df_sensor, df_rest, df_ref, arg_fit):
    # noise + Sensor + reference
    model3 = Model(_2Voigt_gauss_v1)

    # sensor - voigt Fit
    weightV1 = 0.5
    widG1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widL1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    ampG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].max()
    cenG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].idxmax()
    ampL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].max()
    cenL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].idxmax()

    # reference - voigtFit
    weightV2 = 0.5
    ampG2 = df_ref.max()#.values[0]
    cenG2 = df_ref.idxmax()#.values[0]
    widG2 = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2
    ampL2 = df_ref.loc[arg_fit['fit range ref'][0] + widG2:arg_fit['fit range ref'][1]].max()#.values[0]
    cenL2 = df_ref.loc[arg_fit['fit range ref'][0] + widG2:arg_fit['fit range ref'][1]].idxmax()#.values[0]
    widL2 = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2

    # noise - gauss Fit
    wid3 = (df_rest.index[-1] - df_rest.index[0])
    amp3 = df_rest.loc[:wid3 + df_rest.index[0]].max()#.values[0]
    cen3 = df_rest.loc[:wid3 + df_rest.index[0]].idxmax()#.values[0]

    params3 = model3.make_params(weightV1=weightV1, ampG1=ampG1, cenG1=cenG1, widG1=widG1, ampL1=ampL1, cenL1=cenL1,
                                 widL1=widL1, weightV2=weightV2, ampG2=ampG2, cenG2=cenG2, widG2=widG2, ampL2=ampL2,
                                 cenL2=cenL2, widL2=widL2, amp3=amp3, cen3=cen3, wid3=wid3, weight1=1., weight2=0.5,
                                 weight3=0.5)

    # define the frame of the sensor curves
    params3['widG1'].min = 0
    params3['widL1'].min = 0
    params3['ampG1'].min = 0
    params3['ampL1'].min = 0
    params3['cenG1'].min = 0
    params3['cenL1'].min = 0
    params3['amp3'].min = 0
    params3['cen3'].min = 0
    params3['wid3'].min = 0
    params3['weight1'].min = 0
    params3['weight2'].min = 0
    params3['weight3'].min = 0

    # ==========================================
    # lmfit - fitting
    df_tofit = pd.concat([df_sensor, df_ref, df_rest], axis=0).mean(axis=1)
    df_tofit = df_tofit.sort_index()
    result_ = model3.fit(df_tofit.values.tolist(), params3, x=np.array(df_tofit.index), nan_policy='omit')
    df_result = pd.DataFrame(result_.best_fit, index=df_tofit.index)

    return df_result, result_, df_tofit


def optimize_3peaks_v2(df_sensor, df_rest, df_ref, arg_fit):
    # noise + Sensor + reference
    model3 = Model(_2Voigt_gauss_v1)

    # sensor - voigt Fit
    weightV1 = 0.5
    widG1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    widL1 = (df_sensor.index[-1] - df_sensor.index[0]) / 2
    ampG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].max()
    cenG1 = df_sensor.loc[:widG1 + df_sensor.index[0]].idxmax()
    ampL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].max()
    cenL1 = df_sensor.loc[widG1 + df_sensor.index[0] + 30:].idxmax()

    # reference - voigtFit
    weightV2 = 0.5
    ampG2 = df_ref.max()
    cenG2 = df_ref.idxmax()
    widG2 = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2
    ampL2 = df_ref.loc[arg_fit['fit range ref'][0] + widG2:arg_fit['fit range ref'][1]].max()
    cenL2 = df_ref.loc[arg_fit['fit range ref'][0] + widG2:arg_fit['fit range ref'][1]].idxmax()
    widL2 = (arg_fit['fit range ref'][1] - arg_fit['fit range ref'][0]) / 2

    # noise - gauss Fit
    wid3 = (df_rest.index[-1] - df_rest.index[0])
    amp3 = df_rest.loc[:wid3 + df_rest.index[0]].max()
    cen3 = df_rest.loc[:wid3 + df_rest.index[0]].idxmax()

    params3 = model3.make_params(weightV1=weightV1, ampG1=ampG1, cenG1=cenG1, widG1=widG1, ampL1=ampL1, cenL1=cenL1,
                                 widL1=widL1, weightV2=weightV2, ampG2=ampG2, cenG2=cenG2, widG2=widG2, ampL2=ampL2,
                                 cenL2=cenL2, widL2=widL2, amp3=amp3, cen3=cen3, wid3=wid3, weight1=1., weight2=0.5,
                                 weight3=0.5)

    # define the frame of the sensor curves
    params3['widG1'].min = 0
    params3['widL1'].min = 0
    params3['ampG1'].min = 0
    params3['ampL1'].min = 0
    params3['cenG1'].min = 0
    params3['cenL1'].min = 0
    params3['ampG2'].min = 0
    params3['ampL2'].min = 0
    params3['cenG2'].min = 0
    params3['cenL2'].min = 0
    params3['widG2'].min = 0
    params3['widL2'].min = 0
    params3['widG2'].max = 100
    params3['widL2'].max = 100
    params3['amp3'].min = 0
    params3['cen3'].min = 0
    params3['wid3'].min = 0
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


def _curve_fitting_multipleSensors(para, conc, arg, arg_fit, itime, what_fit, save_op, df_rest=None, df_ref=None,
                                   df_Pd=None, df_Pt=None, eval_strategy=None, file_hdr=None, plot_res=True,
                                   save_res=False, save_figure=False):
    print('-----------------------------------------------------------')
    print('Curve fitting for \n', df_Pt.name, ' and ', df_Pd.name)
    if df_Pd is None:
        if df_Pt is None:
            raise ValueError('At least one sensor is required for fitting!')
        arg_fit['ampG'] = df_Pt.loc[arg_fit['fit range Pt'][0]:arg_fit['fit range Pt'][1]].max()
        arg_fit['cenG'] = df_Pt.loc[arg_fit['fit range Pt'][0]:arg_fit['fit range Pt'][1]].idxmax()
    else:
        arg_fit['ampG'] = df_Pd.loc[arg_fit['fit range Pd'][0]:arg_fit['fit range Pd'][1]].max()
        arg_fit['cenG'] = df_Pd.loc[arg_fit['fit range Pd'][0]:arg_fit['fit range Pd'][1]].idxmax()
    if df_Pt is None:
        if df_Pd is None:
            raise ValueError('At least one sensor is required for fitting!')
        arg_fit['ampL'] = df_Pd.loc[arg_fit['fit range Pd'][0]:arg_fit['fit range Pd'][1]].max()
        arg_fit['cenL'] = df_Pd.loc[arg_fit['fit range Pd'][0]:arg_fit['fit range Pd'][1]].idxmax()
    else:
        arg_fit['ampL'] = df_Pt.loc[arg_fit['fit range Pt'][0]:arg_fit['fit range Pt'][1]].max()
        arg_fit['cenL'] = df_Pt.loc[arg_fit['fit range Pt'][0]:arg_fit['fit range Pt'][1]].idxmax()
    arg_fit['widG'] = 10.
    arg_fit['widL'] = 10.

    # --------------------------------------------------------------
    fig_fit = plt.figure(figsize=arg['figure size Fit'])
    ax0 = plt.subplot2grid((11, 5), (0, 0), colspan=2, rowspan=2)
    ax0_res = plt.subplot2grid((11, 5), (2, 0), colspan=2)
    ax1 = plt.subplot2grid((11, 5), (4, 0), colspan=2, rowspan=2)
    ax1_res = plt.subplot2grid((11, 5), (6, 0), colspan=2)
    ax2 = plt.subplot2grid((11, 5), (8, 0), colspan=2, rowspan=2)
    ax2_res = plt.subplot2grid((11, 5), (10, 0), colspan=2)

    ax3 = plt.subplot2grid((11, 5), (0, 3), colspan=2, rowspan=2)
    ax3_res = plt.subplot2grid((11, 5), (2, 3), colspan=2)
    ax4 = plt.subplot2grid((11, 5), (4, 3), colspan=2, rowspan=2)
    ax4_res = plt.subplot2grid((11, 5), (6, 3), colspan=2)
    ax5 = plt.subplot2grid((11, 5), (8, 3), colspan=2, rowspan=2)
    ax5_res = plt.subplot2grid((11, 5), (10, 3), colspan=2)

    ax_space1 = plt.subplot2grid((11, 5), (3, 0), colspan=2)
    ax_space2 = plt.subplot2grid((11, 5), (7, 0), colspan=2)
    ax_space1.axis('off')
    ax_space2.axis('off')
    ax_space3 = plt.subplot2grid((11, 5), (0, 2), rowspan=2)
    ax_space4 = plt.subplot2grid((11, 5), (4, 2), rowspan=2)
    ax_space5 = plt.subplot2grid((11, 5), (8, 2), rowspan=2)
    ax_space3.axis('off')
    ax_space4.axis('off')
    ax_space5.axis('off')

    # sensor1: Pt-TPTBP
    ax0.set_title(df_Pt.name, loc='left', fontsize=arg['fontsize Fit'] * 1.1)
    # ---------------------------------------------------------
    # dictionary of individual peaks - reference, rest and sensor
    # keys: 'result sensor/ref/middle', 'toFit sensor/ref/middle' and plot_para containing figure, ax and residuals
    dic_gfit_pt = dict()
    dic_gtofit_pt = dict()
    l_gaus = _curvefit_sensor(df_sensor=df_Pt, arg_fit=arg_fit, lineshape='gaussian', arg=arg, what_fit=what_fit,
                              fig=fig_fit, col_data=arg['color Pt'], ax=ax0, plot_fig=plot_res, df_rest=df_rest,
                              fitting_range=arg_fit['fit range Pt'], ax_res=ax0_res, df_ref=df_ref)

    dic_gfit_pt[df_Pt.name] = l_gaus[0]['result sensor']
    dic_gtofit_pt[df_Pt.name] = l_gaus[1]['toFit sensor']
    if 'reference' in what_fit:
        dic_gfit_pt['reference'] = l_gaus[0]['result ref']
        dic_gtofit_pt['reference'] = l_gaus[1]['toFit ref']
    if 'middle' in what_fit:
        dic_gfit_pt['rest'] = l_gaus[0]['result middle']
        dic_gtofit_pt['rest'] = l_gaus[1]['toFit middle']
    gfig_pt = l_gaus[2]['figure']

    dic_lfit_pt = dict()
    dic_ltofit_pt = dict()
    l_lorentz = _curvefit_sensor(df_sensor=df_Pt, arg_fit=arg_fit, lineshape='lorentzian', arg=arg, what_fit=what_fit,
                                 fig=fig_fit, col_data=arg['color Pt'], ax=ax1, plot_fig=plot_res, ax_res=ax1_res,
                                 df_ref=df_ref, df_rest=df_rest, fitting_range=arg_fit['fit range Pt'])
    dic_lfit_pt[df_Pt.name] = l_lorentz[0]['result sensor']
    dic_ltofit_pt[df_Pt.name] = l_lorentz[1]['toFit sensor']
    if 'reference' in what_fit:
        dic_lfit_pt['reference'] = l_lorentz[0]['result ref']
        dic_ltofit_pt['reference'] = l_lorentz[1]['toFit ref']
    if 'middle' in what_fit:
        dic_lfit_pt['rest'] = l_lorentz[0]['result middle']
        dic_ltofit_pt['rest'] = l_lorentz[1]['toFit middle']
    lfig_pt = l_lorentz[2]['figure']

    dic_vfit_pt = dict()
    dic_vtofit_pt = dict()
    l_voigt = _curvefit_sensor(df_sensor=df_Pt, arg_fit=arg_fit, lineshape='voigt', arg=arg, what_fit=what_fit,
                               fig=fig_fit, col_data=arg['color Pt'], ax=ax2, plot_fig=plot_res, ax_res=ax2_res,
                               fitting_range=arg_fit['fit range Pt'], df_ref=df_ref, df_rest=df_rest)
    dic_vfit_pt[df_Pt.name] = l_voigt[0]['result sensor']
    dic_vtofit_pt[df_Pt.name] = l_voigt[1]['toFit sensor']
    if 'reference' in what_fit:
        dic_vfit_pt['reference'] = l_voigt[0]['result ref']
        dic_vtofit_pt['reference'] = l_voigt[1]['toFit ref']
    if 'middle' in what_fit:
        dic_vfit_pt['rest'] = l_voigt[0]['result middle']
        dic_vtofit_pt['rest'] = l_voigt[1]['toFit middle']
    vfig_pt = l_voigt[2]['figure']

    # ====================================================================================
    # sensor2: Pd-TPTBP
    ax3.set_title(df_Pd.name, loc='left', fontsize=arg['fontsize Fit'] * 1.1)
    # ---------------------------------------------------------
    # dictionary of individual peaks - reference, rest and sensor
    # keys: 'result sensor/ref/middle', 'toFit sensor/ref/middle' and plot_para containing figure, ax and residuals
    dic_gfit_pd = dict()
    dic_gtofit_pd = dict()
    l_gaus = _curvefit_sensor(df_sensor=df_Pd, arg_fit=arg_fit, lineshape='gaussian', arg=arg, what_fit=what_fit,
                              fig=fig_fit, col_data=arg['color Pd'], ax=ax3, plot_fig=plot_res, ax_res=ax3_res,
                              fitting_range=arg_fit['fit range Pd'], df_ref=df_ref, df_rest=df_rest)
    dic_gfit_pd[df_Pd.name] = l_gaus[0]['result sensor']
    dic_gtofit_pd[df_Pd.name] = l_gaus[1]['toFit sensor']
    if 'reference' in what_fit:
        dic_gfit_pd['reference'] = l_gaus[0]['result ref']
        dic_gtofit_pd['reference'] = l_gaus[1]['toFit ref']
    if 'middle' in what_fit:
        dic_gfit_pd['rest'] = l_gaus[0]['result middle']
        dic_gtofit_pd['rest'] = l_gaus[1]['toFit middle']
    gfig_pd = l_gaus[2]['figure']

    dic_lfit_pd = dict()
    dic_ltofit_pd = dict()
    l_lorentz = _curvefit_sensor(df_sensor=df_Pd, arg_fit=arg_fit, lineshape='lorentzian', arg=arg, what_fit=what_fit,
                                 fig=fig_fit, col_data=arg['color Pd'],  ax=ax4, plot_fig=plot_res, ax_res=ax4_res,
                                 df_ref=df_ref, df_rest=df_rest, fitting_range=arg_fit['fit range Pd'])
    dic_lfit_pd[df_Pd.name] = l_lorentz[0]['result sensor']
    dic_ltofit_pd[df_Pd.name] = l_lorentz[1]['toFit sensor']
    if 'reference' in what_fit:
        dic_lfit_pd['reference'] = l_lorentz[0]['result ref']
        dic_ltofit_pd['reference'] = l_lorentz[1]['toFit ref']
    if 'middle' in what_fit:
        dic_lfit_pd['rest'] = l_lorentz[0]['result middle']
        dic_ltofit_pd['rest'] = l_lorentz[1]['toFit middle']
    lfig_pd = l_lorentz[2]['figure']

    dic_vfit_pd = dict()
    dic_vtofit_pd = dict()
    l_voigt = _curvefit_sensor(df_sensor=df_Pd, arg_fit=arg_fit, lineshape='voigt', arg=arg,
                                                  what_fit=what_fit, fig=fig_fit, col_data=arg['color Pd'], ax=ax5,
                                                  plot_fig=plot_res, fitting_range=arg_fit['fit range Pd'],
                                                  ax_res=ax5_res, df_ref=df_ref, df_rest=df_rest)
    dic_vfit_pd[df_Pd.name] = l_voigt[0]['result sensor']
    dic_vtofit_pd[df_Pd.name] = l_voigt[1]['toFit sensor']
    if 'reference' in what_fit:
        dic_vfit_pd['reference'] = l_voigt[0]['result ref']
        dic_vtofit_pd['reference'] = l_voigt[1]['toFit ref']
    if 'middle' in what_fit:
        dic_vfit_pd['rest'] = l_voigt[0]['result middle']
        dic_vtofit_pd['rest'] = l_voigt[1]['toFit middle']
    vfig_pd = l_voigt[2]['figure']

    plt.tight_layout(w_pad=-5.)
    if plot_res is True:
        plt.show()

    # --------------------------------------------------------------
    dic_fitting = dict({'cube': para, 'Pt data': df_Pt, 'Pd data': df_Pd, 'Pt - gaussian': dic_gfit_pt,
                        'Pt - lorentzian': dic_lfit_pt, 'Pt - voigt': dic_vfit_pt, 'figures': [gfig_pt, lfig_pt, vfig_pt],
                        'Pd - gaussian': dic_gfit_pd, 'Pd - voigt': dic_vfit_pd, 'Pd - lorentzian': dic_lfit_pd,
                        'Pd - figures': [gfig_pd, lfig_pd,  vfig_pd]})

    # =================================================================================================================
    # best fit analysis
    if dic_gfit_pt[df_Pt.name].redchi < dic_lfit_pt[df_Pt.name].redchi:
        if dic_gfit_pt[df_Pt.name].redchi < dic_vfit_pt[df_Pt.name].redchi:
            bestFit_pt = dic_gfit_pt
            pt_tofit = dic_gtofit_pt
            str_bestFit_pt = 'gaussian'
        else:
            bestFit_pt = dic_vfit_pt
            pt_tofit = dic_vtofit_pt
            str_bestFit_pt = 'voigt'
    else:
        if dic_lfit_pt[df_Pt.name].redchi < dic_vfit_pt[df_Pt.name].redchi:
            bestFit_pt = dic_lfit_pt
            pt_tofit = dic_ltofit_pt
            str_bestFit_pt = 'lorentzian'
        else:
            bestFit_pt = dic_vfit_pt
            pt_tofit = dic_vtofit_pt
            str_bestFit_pt = 'voigt'

    if dic_gfit_pd[df_Pd.name].redchi < dic_lfit_pd[df_Pd.name].redchi:
        if dic_gfit_pd[df_Pd.name].redchi < dic_vfit_pd[df_Pd.name].redchi:
            bestFit_pd = dic_gfit_pd
            pd_tofit = dic_gtofit_pd
            str_bestFit_pd = 'gaussian'
        else:
            bestFit_pd = dic_vfit_pd
            pd_tofit = dic_vtofit_pd
            str_bestFit_pd = 'voigt'
    else:
        if dic_lfit_pd[df_Pd.name].redchi < dic_vfit_pd[df_Pd.name].redchi:
            bestFit_pd = dic_lfit_pd
            pd_tofit = dic_ltofit_pd
            str_bestFit_pd = 'lorentzian'
        else:
            bestFit_pd = dic_vfit_pd
            pd_tofit = dic_vtofit_pd
            str_bestFit_pd = 'voigt'
    print('\t best fit for Pt-Sensor: ', str_bestFit_pt)
    print('\t best fit for Pd-Sensor: ', str_bestFit_pd)
    dic_bestFit = dict({'concentration': conc, 'Pt': bestFit_pt, 'Pt data': pt_tofit, 'Pt bestFit': str_bestFit_pt,
                        'Pd': bestFit_pd, 'Pd data': pd_tofit, 'Pd bestFit': str_bestFit_pd})

    # =================================================================================================================
    # saving
    save_curve_fitting(pt_name=df_Pt.name, pd_name=df_Pd.name, arg_fit=arg_fit, conc=conc, itime=itime, save_op=save_op,
                       gfit_pt=dic_gfit_pt, lfit_pt=dic_lfit_pt, vfit_pt=dic_vfit_pt, gfit_pd=dic_gfit_pd,
                       lfit_pd=dic_lfit_pd, vfit_pd=dic_vfit_pd, fig_fit=fig_fit, file_hdr=file_hdr, save_res=save_res,
                       eval_strategy=eval_strategy, save_figure=save_figure)

    return dic_fitting, dic_bestFit


def curve_fitting_dual(file_hdr, path_bestFit, name_dyes, sensor_ID, arg, arg_fit, what_fit, save_op, corr_file,
                       pixel=None, eval_strategy='pixel', plot_cube=False, plot_measurement=True, plot_res=True,
                       save_res=False, save_figure=False):

    print('Analyzing: ', file_hdr)
    # --------------------------------------------------------------
    # preparation
    # pre-check parameter constellation
    if (plot_cube or plot_measurement) is True:
        if pixel is None:
            pixel = [[(200, 696), (150, 850), (230, 750)], [(400, 821), (450, 710), (500, 700)],
                     [(800, 778), (760, 700), (850, 870)]]
            # raise ValueError('Define pixel for evaluation')
        if 'ls' not in arg.keys():
            raise ValueError('Define line style for evaluation')
    if pixel is None:
        pixel = [[(200, 696), (150, 850), (230, 750)], [(400, 821), (450, 710), (500, 700)],
                 [(800, 778), (760, 700), (850, 870)]]

    # define required parameter
    conc = file_hdr.split('Sensor_')[1].split('_')[1]
    if conc.startswith('0'):
        conc_num = np.float(conc.split('pc')[0])/10
    else:
        conc_num = np.float(conc.split('pc')[0])

    # --------------------------------------------------------------
    # load cube
    para = corr.load_cube(file_hdr=file_hdr, corr_file=corr_file, rotation=90., plot_cube=False)
    itime = str(int(para['Integration time'])) + 'ms'

    # correction of measurement signal and extracting individual pixels
    plt.ioff()
    [df_pixel, df_pixel_corr, fig_uncorr,
     fig_corr] = hycam.calibration_solution(para=para, pixel=pixel, name_dyes=name_dyes, ls=arg['ls'], arg=arg,
                                            colors=arg['colors'], plotting='both', plot_cube=plot_cube)
    plt.close(fig_corr)
    plt.close(fig_uncorr)

    # =====================================================================================================
    # how to treat the data
    # either smooth individual pixel (savgol_filter + averaging) or load defined areas (done in HSI Studio)
    # --------------------------------------------------------------
    # individual pixel
    if eval_strategy == 'pixel':
        id_pt = pd.DataFrame(np.zeros(shape=(0, 0)))
        id_pd = pd.DataFrame(np.zeros(shape=(0, 0)))

        # Pt-indicator
        check = 'Pt'
        for en, idx in enumerate(name_dyes):
            if idx[:2].lower() == check.lower():
                id_pt.loc[idx, 0] = int(en)

        # Pd-indicator
        check = 'Pd'
        for en, idx in enumerate(name_dyes):
            if idx[:2].lower() == check.lower():
                id_pd.loc[idx, 0] = int(en)

        name_tag = name_dyes.copy()
        for p in id_pt.index.tolist() + id_pd.index.tolist():
            name_tag.remove(p)

        # Pt/Pd-dualindicator
        id_dual = pd.DataFrame(np.zeros(shape=(len(name_tag), 0)), index=name_tag)
        check = name_tag
        for c in range(len(check)):
            for en, idx in enumerate(name_dyes):
                if idx.lower() == check[c].lower():
                    id_dual.loc[idx, 0] = int(en)
        # id_dual = []
        # for en, idx in enumerate(name_dyes):
        #     if idx[:2].lower() == check.lower():
        #         id_dual.append(en)
        dev_name = dict()
        string_dev = dict()
        dict_dual = dict()
        for en, tag in enumerate(id_dual.index):
            dict_dual[tag] = None

        for i in range(len(id_dual.index)):
            dev_name[i] = [li for li in difflib.ndiff(id_dual.index[i], min(id_dual.index, key=len)) if li[0] != ' ']
            if '- +' in dev_name:
                dev_name[i].remove('- +')
            string_dev[i] = ''.join([i.split(' ')[1] for i in dev_name[i]])

        for en in string_dev:
            sensor_id = [x for x in id_dual.index if string_dev[en] in x]

            if len(sensor_id) == 1:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_dual.loc[sensor_id, 0].values[0])]].mean(axis=1), 7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_dual[sensor_id[0]] = df_
            else:
                sm = savgol_filter(df_pixel_corr[pixel[int(id_dual.loc[min(id_dual.index, key=len), 0])]].mean(axis=1),
                                   7, 3)
                df_ = pd.DataFrame(sm, index=df_pixel_corr.index)
                dict_dual[min(id_dual.index, key=len)] = df_
        df_dual = pd.concat(dict_dual, axis=1, sort=False)
        df_dual.columns = [col[0] for col in df_dual.columns]

        # df_dual = pd.DataFrame(savgol_filter(df_pixel_corr[pixel[id_dual[0]]].mean(axis=1), 7, 3),
        #                        index=df_pixel_corr.index)

    # --------------------------------------------------------------
    elif eval_strategy == 'area':
        file_area = '_'.join(file_hdr.split('cube')[0].split('pc')[0].split('_')[:-1]) + '_' + conc + '_spectrum.csv'
        if path.exists(file_area) is True:
            pass
        else:
            file_area = '_'.join(file_hdr.split('cube')[0].split('pc')[0].split('_')[:-1]) + '_' + conc + '_spectra.csv'
            if path.exists(file_area) is True:
                pass
            else:
                raise ValueError('Check path+filename of spectra')

        df_ = pd.read_csv(file_area, header=None).T.loc[4:].set_index(0)
        col_name = pd.read_csv(file_area, header=None).T.loc[0].values[1:]
        df_.columns = col_name
        df_.index.name = None

        # correction of regions
        kappa = para['correction']['correction factor']
        index_new = [round(i, 3) for i in kappa.index]
        kappa.index = index_new
        df_corr = hycam.cube_signal_correction(df_uncorr=df_, kappa=kappa)

        # Pt-indicator
        check = 'Pt'
        id_pt = pd.DataFrame(np.zeros(shape=(0, 0)))
        for en, idx in enumerate(col_name):
            if idx[:2].lower() == check.lower():
                id_pt.loc[idx, 0] = int(en)

        # Pd-indicator
        check = 'Pd'
        id_pd = pd.DataFrame(np.zeros(shape=(0, 0)))
        for en, idx in enumerate(col_name):
            if idx[:2].lower() == check.lower():
                id_pd.loc[idx, 0] = int(en)

        # Pt/Pd-Dualindicator
        check = 'Du'
        name_tag = name_dyes.copy()
        for p in id_pt.index.tolist() + id_pd.index.tolist():
            name_tag.remove(p)
        id_dual = pd.DataFrame(np.zeros(shape=(len(name_tag), 0)), index=name_tag)
        for en, idx in enumerate(col_name):
            if idx[:2].lower() == check.lower():
                id_dual.loc[idx, 0] = int(en)

        dev_name = dict()
        string_dev = dict()
        dict_dual = dict()
        for en, tag in enumerate(id_dual.index):
            dict_dual[tag] = None

        for i in range(len(id_dual.index)):
            dev_name[i] = [li for li in difflib.ndiff(id_dual.index[i], min(id_dual.index, key=len)) if li[0] != ' ']
            if '- +' in dev_name:
                dev_name[i].remove('- +')
            string_dev[i] = ''.join([i.split(' ')[1] for i in dev_name[i]])

        for en in string_dev:
            sensor_id = [x for x in id_dual.index if string_dev[en] in x]
            if len(sensor_id) == 1:
                df_ = pd.DataFrame(df_corr[sensor_id])
                dict_dual[sensor_id[0]] = df_
            else:
                df_ = pd.DataFrame(df_corr[min(sensor_id, key=len)])
                dict_dual[min(id_dual.index, key=len)] = df_

    # --------------------------------------------------------------
    else:
        raise ValueError('Define evaluation strategy, whether individual pixel or a whole (pre-defined) region should '
                         'be analyzed: eval_strategy = pixel /area')

    # --------------------------------------------------------------
    # Plot measurement
    if plot_measurement is True:
        fig, ax = plot.plotting_dualoptodes(df_dual=dict_dual, arg=arg)
        plt.show()
    else:
        plt.ioff()
        fig, ax = plot.plotting_dualoptodes(df_dual=dict_dual, arg=arg)
        plt.close(fig)

    # --------------------------------------------------------------------
    # load best fit for individual sensors
    s_fit_range_pd = str(arg_fit['fit range Pd'][0]) + '-' + str(arg_fit['fit range Pd'][1])
    s_fit_range_pt = str(arg_fit['fit range Pt'][0]) + '-' + str(arg_fit['fit range Pt'][1])

    # Curve fitting_area_Pt-TPTBP+MY-700-900-Pd-TPTBP+MY-700-900_Pd-TPTBP+MY
    if eval_strategy == 'pixel':
        file_Pd = path_bestFit + '/pixel/' + conc + '/' + 'Pd-' + sensor_ID.split('-')[1] + '_pixel_' + \
                  arg_fit['ls Pd'].loc[conc_num].values[0] + '-fit_' + s_fit_range_pd + '_sensor.txt'
        file_Pt = path_bestFit + '/pixel/' + conc + '/' + sensor_ID + '_pixel_' + \
                  arg_fit['ls Pt'].loc[conc_num].values[0] + '-fit_' + s_fit_range_pt + '_sensor.txt'
    elif eval_strategy == 'area':
        file_Pd = path_bestFit + '/area/' + conc + '/' + 'Curvefitting_area_' + sensor_ID + '-' + s_fit_range_pt + '-'\
                  + 'Pd-' + sensor_ID.split('-')[1] + '-' + s_fit_range_pd + '_' + 'Pd-' + sensor_ID.split('-')[1] + \
                  '.txt'
        file_Pt = path_bestFit + '/area/' + conc + '/' + 'Curvefitting_area_' + sensor_ID + '-' + s_fit_range_pt + '-'\
                  + 'Pd-' + sensor_ID.split('-')[1] + '-' + s_fit_range_pd + '_' + sensor_ID + '.txt'

    if isinstance(file_Pt, str):
        pass
    else:
        file_Pt = file_Pt[0]
        file_Pd = file_Pd[0]

    # --------------------------------------------------------------------
    # general loading of fit results
    df_en = pd.DataFrame(np.zeros(shape=(5, 2)), index=['Fit Statistics', '##', 'Variables', 'Correlations', 'End'],
                         columns=['Pd', 'Pt'])
    with open(file_Pd, 'r') as file:
        for en, l in enumerate(file):
            if l.startswith('[[Fit Statistics]]'):
                df_en.loc['Fit Statistics', 'Pd'] = en
            if l.startswith('##'):
                df_en.loc['##', 'Pd'] = en
            if l.startswith('[[Variables]]'):
                df_en.loc['Variables', 'Pd'] = en
            if l.startswith('[[Correlations]]'):
                df_en.loc['Correlations', 'Pd'] = en
        df_en.loc['End', 'Pd'] = en

    with open(file_Pt, 'r') as file:
        for en, l in enumerate(file):
            if l.startswith('[[Fit Statistics]]'):
                df_en.loc['Fit Statistics', 'Pt'] = en
            if l.startswith('##'):
                df_en.loc['##', 'Pt'] = en
            if l.startswith('[[Variables]]'):
                df_en.loc['Variables', 'Pt'] = en
            if l.startswith('[[Correlations]]'):
                df_en.loc['Correlations', 'Pt'] = en
        df_en.loc['End', 'Pt'] = en

    # --------------------------------------------------------------------
    # statistics included
    df_statistics = pd.DataFrame(np.zeros(shape=(1, 2)), index=['# fitting method'], columns=['Pd', 'Pt'])
    if df_en.loc['##', 'Pd'] == 0:
        foot_pd = df_en.loc['End', 'Pd'] - df_en.loc['Variables', 'Pd'] + df_en.loc['Fit Statistics', 'Pd']
    else:
        foot_pd = df_en.loc['End', 'Pd'] - df_en.loc['##', 'Pd'] + df_en.loc['Fit Statistics', 'Pd']

    l_pd = pd.read_csv(file_Pd, skip_blank_lines=True, skiprows=int(df_en.loc['Fit Statistics', 'Pd']),
                       skipfooter=int(foot_pd), engine='python').values
    for i in range(len(l_pd)):
        df_statistics.loc[l_pd[i][0].split('=')[0].strip(), 'Pd'] = l_pd[i][0].strip().split('=')[1]

    if df_en.loc['##', 'Pt'] == 0:
        foot_pt = df_en.loc['End', 'Pt'] - df_en.loc['Variables', 'Pt'] + df_en.loc['Fit Statistics', 'Pt']
    else:
        foot_pt = df_en.loc['End', 'Pt'] - df_en.loc['##', 'Pt'] + df_en.loc['Fit Statistics', 'Pt']
    l_pt = pd.read_csv(file_Pt, skip_blank_lines=True, skiprows=int(df_en.loc['Fit Statistics', 'Pt']),
                       skipfooter=int(foot_pt), engine='python').values
    for i in range(len(l_pt)):
        df_statistics.loc[l_pt[i][0].split('=')[0].strip(), 'Pt'] = l_pt[i][0].split('=')[1].strip()

    # --------------------------------------------------------------------
    # parameters
    if df_en.loc['Correlations', 'Pd'] == 0:
        foot_pd = 0
    else:
        foot_pd = df_en.loc['End', 'Pd'] - df_en.loc['Correlations', 'Pd'] + 1

    df_params = pd.DataFrame(np.zeros(shape=(1, 2)), columns=['Pd', 'Pt'])
    l_pd = pd.read_csv(file_Pd, skip_blank_lines=True, skiprows=int(df_en.loc['Variables', 'Pd']),
                       skipfooter=int(foot_pd), engine='python').values
    for i in range(len(l_pd)):
        df_params.loc[l_pd[i][0].strip().split(':')[0].strip(), 'Pd'] = np.float(
            l_pd[i][0].strip().split(':')[1].split('+')[0].strip().split(' ')[0])

    if df_en.loc['Correlations', 'Pt'] == 0:
        foot_pt = 0
    else:
        foot_pt = df_en.loc['End', 'Pt'] - df_en.loc['Correlations', 'Pt'] + 1

    l_pt = pd.read_csv(file_Pt, skip_blank_lines=True, skiprows=int(df_en.loc['Variables', 'Pt']),
                       skipfooter=int(foot_pt), engine='python').values
    for i in range(len(l_pt)):
        df_params.loc[l_pt[i][0].strip().split(':')[0].strip(), 'Pt'] = np.float(
            l_pt[i][0].strip().split(':')[1].split('+')[0].strip().split(' ')[0])
    df_params = df_params.drop(0)

    # ======================================================================
    # model for lmfit
    if isinstance(arg_fit['ls dual'].loc[conc_num].values[0], str):
        print('\t', 'lineshape dualindicator: ', arg_fit['ls dual'].loc[conc_num].values[0])
    else:
        print('\t', 'lineshape dualindicator: ', arg_fit['ls dual'].loc[conc_num].values[0][0])

    dic_result_sensor = dict()
    dic_result_ref = dict()
    dic_result_rest = dict()
    dic_bestFit_sensor = dict()
    dic_bestFit_ref = dict()
    dic_bestFit_rest = dict()
    dic_tofit_sensor = dict()
    dic_tofit_ref = dict()
    dic_tofit_rest = dict()
    dic_plot_all = dict()

    for en, c in enumerate(dict_dual.keys()):
        plt.ioff()
        fig_fit = plt.figure(figsize=arg['figure size'])
        ax0 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        ax0_res = plt.subplot2grid((4, 4), (2, 0), colspan=2)
        ax0.set_title(c, loc='left', fontsize=arg['fontsize Fit'])

        # ---------------------------------------------------------
        # splitting function into parts
        df_sensor = pd.DataFrame(dict_dual[c][c].loc[arg_fit['fit range dual'][0]:arg_fit['fit range dual'][1]]).astype(float)

        what_fit_ = what_fit.copy()
        if max(string_dev.values(), key=len) in c:
            if 'reference' in what_fit_:
                pass
            else:
                what_fit_.append('reference')
            df_ref = pd.DataFrame(dict_dual[c][c].loc[arg_fit['fit range ref'][0]:arg_fit['fit range ref'][1]].astype(float))
        else:
            if 'reference' in what_fit_:
                what_fit_.remove('reference')
            df_ref = None
        if 'middle' in what_fit_:
            df_rest = pd.DataFrame(dict_dual[c][c].loc[arg_fit['fit range ref'][1]:arg_fit['fit range dual'][0]].astype(float))
        else:
            df_rest = None

        # ---------------------------------------------------------
        if df_ref is None:
            df_result, result_, df_tofit = optimize_2peaks(df_sensor=df_sensor.loc[:, df_sensor.columns[0]],
                                                           df_rest=df_rest)
        else:
            df_result, result_, df_tofit = optimize_3peaks(df_sensor=df_sensor.loc[:, df_sensor.columns[0]],
                                                           df_rest=df_rest, df_ref=df_ref, arg_fit=arg_fit)

        ax0.plot(df_tofit, marker='o', fillstyle='none', lw=0, color=arg['color dual 1'], label='data toFit')
        ax0.plot(df_result, ls='-.', color='k', label='Fit {:.2e}'.format(result_.chisqr))
        ax0_res.axhline(0, lw=0.5, color='k')
        ax0_res.plot(df_result[0] - df_tofit, marker='.', lw=0, color='k')

        ax0.legend(frameon=True, fancybox=True, loc=0)
        ax0.tick_params(axis='x', labelsize=0)

        # dictionary of individual peaks - reference, rest and sensor
        # keys: 'result sensor/ref/middle', 'toFit sensor/ref/middle' and plot_para containing figure, ax and residuals
        # l = _curvefit_sensor(df_sensor=dict_dual[c][c], df_ref=df_ref, df_rest=df_rest, arg_fit=arg_fit, arg=arg,
        #                      ax_res=ax0_res, plot_fig=plot_res, col_data=arg['color dual 1'], fig=fig_fit, ax=ax0,
        #                      fitting_range=arg_fit['fit range dual'],  df_paraD=df_params, what_fit=what_fit_,
        #                      lineshape=arg_fit['ls dual'].loc[conc_num].values[0])
        # plt.tight_layout(w_pad=2)
        if plot_res is True:
            plt.show()
        else:
            plt.close(fig_fit)

        dic_result_sensor[c] = result_ # l[0]['result sensor']
        dic_bestFit_sensor[c] = df_result # pd.DataFrame(l[0]['result sensor'].best_fit, index=l[1]['toFit sensor'].index)
        dic_tofit_sensor[c] = df_tofit # l[1]['toFit sensor']

        if 'reference' in what_fit_:
            dic_tofit_ref[c] = df_ref
        #     dic_result_ref[c] = l[0]['result ref']
        #     dic_bestFit_ref[c] = pd.DataFrame(l[0]['result ref'].best_fit, index=l[1]['toFit ref'].index)

        if 'middle' in what_fit_:
            dic_tofit_rest[c] = df_rest
        #     dic_result_rest[c] = l[0]['result middle']
        #     dic_bestFit_rest[c] = pd.DataFrame(l[0]['result middle'].best_fit, index=l[1]['toFit middle'].index)

        # dic_plot_all[c] = l[2]['figure']
        # print(1940)
        # ======================================================================
        # saving
        save_curve_fitting(arg_fit=arg_fit, conc=conc, itime=itime, save_op=save_op, dual_name=c, fit_dual=l[0],
                           fig_fit=fig_fit, file_hdr=file_hdr, eval_strategy=eval_strategy, save_res=save_res,
                           save_figure=save_figure)

    # --------------------------------------------------------------
    dic_fitting = dict({'cube': para, 'concentration': conc_num, 'data': dict_dual, 'fit sensor': dic_result_sensor,
                         'figures': dic_plot_all, 'data toFit rest': dic_tofit_rest,  'parameter individuals': df_params,
                        'ls dual': arg_fit['ls dual'].loc[conc_num].values[0], 'data toFit sensor': dic_tofit_sensor,
                        'data toFit ref': dic_tofit_ref, 'bestFit sensor': dic_bestFit_sensor})
                        # 'fit ref': dic_result_ref, 'fit middle': dic_result_rest, 'bestFit ref': dic_bestFit_ref,
    #                         'bestFit middle': dic_bestFit_rest,

    return dic_fitting


def signal_deconvolution(df, arg, eval_strategy, arg_fit, save_op, path_save=None, plotting=True, save_res=False,
                         save_figure=False):
    conc_ = df['concentration']
    print('Concentration analyzed: ', conc_, '% O_2')
    x = df['data toFit'].index
    params_ind = df['parameter individuals']  # from individual fit

    # ====================================================================================================
    # Pt first, then Pd
    y_pt = _1Voigt_v1(x=x, weightG=params_ind['Pt'].loc['weightG'], weightL=params_ind['Pt'].loc['weightL'],
                      ampG=params_ind['Pt'].loc['ampG'], cenG=params_ind['Pt'].loc['cenG'],
                      widG=params_ind['Pt'].loc['widG'], ampL=params_ind['Pt'].loc['ampL'],
                      cenL=params_ind['Pt'].loc['cenL'], widL=params_ind['Pt'].loc['widL'])
    y_pd = _1Voigt_v1(x=x, weightG=params_ind['Pd'].loc['weightG'], weightL=params_ind['Pd'].loc['weightL'],
                      ampG=params_ind['Pd'].loc['ampG'], cenG=params_ind['Pd'].loc['cenG'],
                      widG=params_ind['Pd'].loc['widG'], ampL=params_ind['Pd'].loc['ampL'],
                      cenL=params_ind['Pd'].loc['cenL'], widL=params_ind['Pd'].loc['widL'])

    df_individual = pd.DataFrame([y_pt, y_pd], columns=x, index=['Pt', 'Pd']).T

    # construction of dual-indicator curve (fitted based on measurement)
    df_dual = df['data toFit']
    df_dual.columns = ['dualFit']
    df_data = pd.concat([df_individual, df_dual], axis=1, sort=False)

    # ----------------------------------------------------------------------------------------------------------
    # lmfit -> fit weighting factors for sum
    smodel = Model(_2Voigt_v1)
    para_sum = smodel.make_params(weight1=0.1, weight2=0.9, weightG1=params_ind['Pt'].loc['weightG'],
                                  ampG1=params_ind['Pt'].loc['ampG'], weightL1=params_ind['Pt'].loc['weightL'],
                                  cenG1=params_ind['Pt'].loc['cenG'], widG1=params_ind['Pt'].loc['widG'],
                                  ampL1=params_ind['Pt'].loc['ampL'], cenL1=params_ind['Pt'].loc['cenL'],
                                  widL1=params_ind['Pt'].loc['widL'], weightG2=params_ind['Pd'].loc['weightG'],
                                  weightL2=params_ind['Pd'].loc['weightL'], ampG2=params_ind['Pd'].loc['ampG'],
                                  cenG2=params_ind['Pd'].loc['cenG'], widG2=params_ind['Pd'].loc['widG'],
                                  ampL2=params_ind['Pd'].loc['ampL'], cenL2=params_ind['Pd'].loc['cenL'],
                                  widL2=params_ind['Pd'].loc['widL'])

    # only weight1 and weight2 are variable!!!!!
    for k in para_sum.keys():
        if k == 'weight1':
            pass
        elif k == 'weight2':
            pass
        else:
            para_sum[k].vary = False

    # --------------------------------------------
    # fit sum of individual sensors to get superimposed signal
    df_tofit_dual = df_dual.loc[arg_fit['fit range dual'][0]:arg_fit['fit range dual'][1]]
    result_sum = smodel.fit(df_tofit_dual['dualFit'].values, para_sum, x=np.array(df_tofit_dual.index),
                            nan_policy='omit')

    # ==================================================================
    # plotting signal deconvolution results
    plt.ioff()
    fig, ax = plt.subplots(figsize=arg['figure size'])
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    for c in df_data.columns:
        label_col = 'color ' + c.split('_')[0].split('F')[0]
        if c == 'dualFit':
            ax.plot(df_data[c], color=arg[label_col], lw=1.75, ls='--', label=c)
        else:
            ax.plot(df_data[c], color=arg[label_col], lw=1.75, label=c)

    ax.plot(np.array(df_tofit_dual.index), result_sum.best_fit, color=arg[label_col], marker='.', lw=0.,
            label='sum χ2 = {:.2e}'.format(result_sum.redchi))

    ax.legend(loc=0, fontsize=arg['fontsize']*0.8)
    ax.set_xlabel('Wavelength [nm]', fontsize=arg['fontsize'])
    ax.set_ylabel('Intensity [rfu]', fontsize=arg['fontsize'])

    ax.tick_params(axis='both', which='both', direction='out', labelsize=arg['fontsize']*0.9)
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)
    plt.tight_layout()

    if plotting is True:
        plt.show()
    else:
        plt.close(fig)

    # =======================================================================================
    # saving
    if (save_res or save_figure) is True:
        if 'type' in save_op.keys():
            fig_type = save_op['type']
        else:
            fig_type = 'png'
        if 'dpi' in save_op:
            dpi = save_op['dpi']
        else:
            dpi = 300.
        if path_save is None:
            raise ValueError('Directory to save results is required')

        # create folder if it doesn't exist
        if conc_ == 0.:
            conc_str = '0pc'
        elif conc_ < 1.:
            conc_str = str(0) + str(conc_).split('.')[1] + 'pc'
        else:
            conc_str = str(int(conc_)) + 'pc'
        if eval_strategy == 'pixel':
            file_path = path_save + '/pixel/' + conc_str
        else:
            file_path = path_save + '/area/' + conc_str
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        # define saving name
        if eval_strategy == 'pixel':
            fig_name = 'CurveDecon_pixel_' + str(arg_fit['fit range dual'][0]) + '-' + str(arg_fit['fit range dual'][1]) + '.'
            fitname = 'Dual-TPTBP_pixel_decon_' + str(arg_fit['fit range dual'][0]) + '-' + str(arg_fit['fit range dual'][1]) + '.'

        else:
            fig_name = 'CurveDecon_area_' + str(arg_fit['fit range dual'][0]) + '-' + str(arg_fit['fit range dual'][1]) + '.'
            fitname = 'Pt-TPTBP_area_decon_' + str(arg_fit['fit range dual'][0]) + '-' + str(arg_fit['fit range dual'][1]) + '.'

        # save fitting results to folder
        if save_res is True:
            with open(file_path + '/' + fitname + 'txt', 'w') as fh:
                fh.write(result_sum.fit_report())

        # save figure to folder
        if save_figure is True:
            fig.savefig(file_path + '/' + fig_name + fig_type, dpi=dpi)

    # =======================================================================================
    sum_param = dict({'data': df_data, 'conc': conc_, 'Fit': result_sum, 'Pt alpha': result_sum.params['weight1'].value,
                      'Pd alpha': result_sum.params['weight2'].value, 'figure': fig, 'ax': ax})

    return sum_param

