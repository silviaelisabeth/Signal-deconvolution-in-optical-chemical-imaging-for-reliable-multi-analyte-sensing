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
from matplotlib import cm
import matplotlib.patches as patches
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from scipy import ndimage
import pandas as pd
import numpy as np
from lmfit import Model
from spectral import *
import scipy.integrate as integrate

import layout_plotting_v1_3 as plot
import curve_fitting_v1_1 as curve


# =====================================================================================
def cube_signal_correction(df_uncorr, kappa):
    for col in df_uncorr.columns:
        index_new = [round(i, 3) for i in df_uncorr[col].index]
        df_uncorr[col].index = index_new

    df_corr = df_uncorr.copy()
    for col in df_uncorr.columns:
        print(col)
        df_corr_crop = kappa.loc[int(df_uncorr[col].index[0]): df_uncorr[col].index[-1]]
        l = df_uncorr[col] * df_corr_crop
        print(l)
        df_corr.loc[:, col] = l.values

    return df_corr


def plotting_cube(img_cube, pixel, span_area, arg=None):
    if arg is None:
        px_color = 'k'
        rotation = 0.
        cmap_ = 'jet'
    else:
        if 'px color' in arg.keys():
            px_color = arg['px color']
        else:
            px_color = 'k'

        if 'rotation' in arg.keys():
            rotation = arg['rotation']
        else:
            rotation = 0.
        if 'cmap' in arg.keys():
            cmap_ = str(arg['cmap'])
        else:
            cmap_ = 'jet'

    img = img_cube.open_memmap()
    if rotation != 0:
        img_rot = ndimage.rotate(img, rotation)
        imshow(img_rot, cmap=cmap_)
    else:
        imshow(img, cmap=cmap_)

    if span_area == 'True':
        if 'facecolor' in arg.keys():
            facecol = arg['facecolor']
        else:
            facecol = 'darkorange'
        if 'alpha area' in arg.keys():
            alpha_ = arg['alpha area']
        else:
            alpha_ = 0.25

        if arg['rotation'] == 0.:
            plt.axes().add_patch(patches.Rectangle(pixel[0],  # (x,y)
                pixel[2][0] - pixel[0][0],  #  width
                pixel[1][1] - pixel[0][1],  #  height
                facecolor=facecol, alpha=alpha_))
        elif arg['rotation'] == 90.:
            plt.axes().add_patch(patches.Rectangle(pixel[0],  # (x,y)
                pixel[1][1] - pixel[0][1], #  width
                pixel[2][0] - pixel[0][0], #  height
                facecolor=facecol, alpha=alpha_))
    else:
        for pi in pixel:
            if arg['rotation'] == 90.:
                plt.annotate(pi, xy=(pi[0], pi[1]), xycoords='data', xytext=(7, 10), textcoords='offset points',
                             color=px_color, arrowprops=dict(arrowstyle="->", color=px_color, connectionstyle="arc3"))
            elif arg['rotation'] == 0.:
                plt.annotate(pi, xy=(pi[1], pi[0]), xycoords='data', xytext=(7, 10), textcoords='offset points',
                             color=px_color, arrowprops=dict(arrowstyle="->", color=px_color, connectionstyle="arc3"))
    return


def plotting_spectra_from_pixel(df_pixel, pixel, name_indicator, colors, ls, figsize=(12, 4), fontsize_=13,
                                corrected=True):

    # make it more flexible -> number of subplots according to number of different indicators/sensors
    num_subplots = len(name_indicator)
    fig, ax = plt.subplots(ncols=num_subplots, figsize=figsize, sharex=True, sharey=True)

    if corrected is True:
        fig.suptitle('Corrected spectra', fontsize=fontsize_)
    else:
        fig.suptitle('Uncorrected spectra', fontsize=fontsize_)
    for en, n in enumerate(name_indicator):
        ax[en].set_title(n, loc='left', fontsize=fontsize_ * 0.9)

    i = 0.
    for k, t in enumerate(pixel):
        for en, pi in enumerate(t):
            ax[k].plot(df_pixel[pi], lw=1.5, ls=ls[int(i % 3)], color=colors(int(i)), label=pi)
            i += 1
    for en in range(num_subplots):
        ax[en].legend(frameon=True, fancybox=True, loc=0, fontsize=fontsize_*0.7)
        ax[en].tick_params(axis='both', which='both', direction='out', labelsize=fontsize_ * 0.9)

    for en in range(num_subplots):
        ax[en].set_xlabel('Wavelength [nm]', fontsize=fontsize_)
    ax[0].set_ylabel('Intensity', fontsize=fontsize_)
    plt.tight_layout(pad=3)

    return fig, ax


def calibration_solution(para, pixel, name_dyes, ls, span_area='False', plot_cube=True, colors='viridis', arg=None,
                         plotting='both'):
    # plotting cube
    ls_pixel = pixel[0].copy()
    for i in range(len(pixel) - 1):
        ls_pixel += pixel[i + 1]

    if plot_cube is True:
        plotting_cube(img_cube=para['cube'], pixel=ls_pixel, span_area=span_area, arg=arg)

    # --------------------------------------------------------
    # create dictionary for all pixels
    dic_pixel = {}
    for p in pixel:
        for pi in p:
            dic_pixel[pi] = para['cube'].open_memmap()[pi[0]][pi[1]]
    df_pixel = pd.DataFrame.from_dict(dic_pixel, orient='index', columns=para['Wavelength']).T.sort_index().drop(0.)

    # --------------------------------------------------------
    # plot (un)corrected spectra
    if arg is None:
        figsize_ = (12, 4)
        fontsize_ = 13
    else:
        if 'figure size' in arg.keys():
            figsize_ = arg['figure size']
        else:
            figsize_ = (12, 4)
        if 'fontsize' in arg.keys():
            fontsize_ = arg['fontsize']
        else:
            fontsize_ = 13

    # uncorrected spectra -> raw data
    ls_colors = cm.get_cmap(colors, len(df_pixel.columns))

    # ------------------------------
    # load correction file
    df_corr = para['correction']
    df_pixel_corr = cube_signal_correction(df_uncorr=df_pixel, kappa=df_corr['correction factor'])

    if plotting == 'both':
        fig_uncorr, ax_uncorr = plotting_spectra_from_pixel(df_pixel=df_pixel, pixel=pixel, name_indicator=name_dyes,
                                                            colors=ls_colors, corrected=False, figsize=figsize_,
                                                            fontsize_=fontsize_, ls=ls)
        fig_corr, ax_corr = plotting_spectra_from_pixel(df_pixel=df_pixel_corr, pixel=pixel, name_indicator=name_dyes,
                                                        colors=ls_colors, corrected=True, figsize=figsize_,
                                                        fontsize_=fontsize_, ls=ls)
    elif plotting == 'uncorrected':
        fig_uncorr, ax_uncorr = plotting_spectra_from_pixel(df_pixel=df_pixel, pixel=pixel, name_indicator=name_dyes,
                                                            colors=ls_colors, corrected=False, figsize=figsize_,
                                                            fontsize_=fontsize_, ls=ls)
        fig_corr = None
    elif plotting == 'corrected':
        fig_corr, ax_corr = plotting_spectra_from_pixel(df_pixel=df_pixel_corr, pixel=pixel, name_indicator=name_dyes,
                                                        colors=ls_colors, corrected=True, figsize=figsize_,
                                                        fontsize_=fontsize_, ls=ls)
        fig_uncorr = None
    elif plotting is False:
        fig_uncorr = None
        fig_corr = None
    else:
        raise ValueError('Choose for plotting either (1) uncorrected or (2) corrected spectra.')

    return df_pixel, df_pixel_corr, fig_uncorr, fig_corr


def calibration_point(file_hdr, dic_fitting, fitrange_Pt, fitrange_Pd, info=True):
    if info is True:
        print('load: ', file_hdr)
    conc = file_hdr.split('Sensor_')[1].split('_')[1]
    if conc.startswith('0'):
        conc_num = int(conc.split('pc')[0]) / 10
    else:
        conc_num = np.float(conc.split('pc')[0])
    slit = np.float(dic_fitting['cube']['cube'].metadata['pixel step'])

    # crop data to fitting range
    df_Pt = pd.DataFrame(dic_fitting['Pt data'].loc[fitrange_Pt[0]:fitrange_Pt[1]])
    df_Pd = pd.DataFrame(dic_fitting['Pd data'].loc[fitrange_Pd[0]:fitrange_Pd[1]])
    xmax_pt = df_Pt.idxmax().values[0]
    xmax_pd = df_Pd.idxmax().values[0]

    ymax_pt = pd.DataFrame(dic_fitting['Pt data']).loc[xmax_pt - slit - 0.5: xmax_pt + slit + 0.5].mean().values[0]
    ymax_pt_std = pd.DataFrame(dic_fitting['Pt data']).loc[xmax_pt - slit - 0.5: xmax_pt + slit + 0.5].std().values[0]
    ymax_pd = pd.DataFrame(dic_fitting['Pd data']).loc[xmax_pd - slit - 0.5: xmax_pd + slit + 0.5].mean().values[0]
    ymax_pd_std = pd.DataFrame(dic_fitting['Pd data']).loc[xmax_pd - slit - 0.5: xmax_pd + slit + 0.5].std().values[0]

    # combine information in dictionary
    calib_point = dict({'concentration [%]': conc_num, 'wavelength Pt [nm]': xmax_pt, 'intensity Pt [rfu]': ymax_pt,
                        'std Pt [rfu]': ymax_pt_std, 'wavelength Pd [nm]': xmax_pd, 'intensity Pd [rfu]': ymax_pd,
                        'std Pd [rfu]': ymax_pd_std})

    return calib_point


def calibration_point_dual(file_hdr, dic_fitting, info=True):
    if info is True:
        print('load: ', file_hdr)
    conc = file_hdr.split('Sensor_')[1].split('_')[1]
    if conc.startswith('0'):
        conc_num = int(conc.split('pc')[0]) / 10
    else:
        conc_num = np.float(conc.split('pc')[0])
    slit = np.float(dic_fitting['cube']['cube'].metadata['pixel step'])

    # reconstruction of individual sensors
    params_ind = dic_fitting['parameter individuals']
    x = dic_fitting['data toFit'].index
    y_pt = curve._1Voigt_v1(x=x, weightG=params_ind['Pt'].loc['weightG'], weightL=params_ind['Pt'].loc['weightL'],
                            ampG=params_ind['Pt'].loc['ampG'], cenG=params_ind['Pt'].loc['cenG'],
                            widG=params_ind['Pt'].loc['widG'], ampL=params_ind['Pt'].loc['ampL'],
                            cenL=params_ind['Pt'].loc['cenL'], widL=params_ind['Pt'].loc['widL'])
    y_pd = curve._1Voigt_v1(x=x, weightG=params_ind['Pd'].loc['weightG'], weightL=params_ind['Pd'].loc['weightL'],
                            ampG=params_ind['Pd'].loc['ampG'], cenG=params_ind['Pd'].loc['cenG'],
                            widG=params_ind['Pd'].loc['widG'], ampL=params_ind['Pd'].loc['ampL'],
                            cenL=params_ind['Pd'].loc['cenL'], widL=params_ind['Pd'].loc['widL'])

    df_individual = pd.DataFrame([y_pt, y_pd], columns=x, index=['Pt', 'Pd']).T
    xmax_pt = df_individual.idxmax()['Pt']
    xmax_pd = df_individual.idxmax()['Pd']

    ymax_pt = df_individual['Pt'].loc[xmax_pt - slit - 0.5: xmax_pt + slit + 0.5].mean()
    ymax_pt_std = df_individual['Pt'].loc[xmax_pt - slit - 0.5: xmax_pt + slit + 0.5].std()
    ymax_pd = df_individual['Pd'].loc[xmax_pd - slit - 0.5: xmax_pd + slit + 0.5].mean()
    ymax_pd_std = df_individual['Pd'].loc[xmax_pd - slit - 0.5: xmax_pd + slit + 0.5].std()

    # combine information in dictionary
    calib_point = dict({'concentration [%]': conc_num, 'wavelength Pt [nm]': xmax_pt, 'intensity Pt [rfu]': ymax_pt,
                        'std Pt [rfu]': ymax_pt_std, 'wavelength Pd [nm]': xmax_pd, 'intensity Pd [rfu]': ymax_pd,
                        'std Pd [rfu]': ymax_pd_std})

    return calib_point


def integral_calibration_point(file_hdr, best_fit, sensor_id, a, lineshape=None, limit_integral=(650, 900)):
    conc = file_hdr.split('Sensor_')[1].split('_')[1]
    if conc.startswith('0'):
        conc_num = int(conc.split('pc')[0]) / 10
    else:
        conc_num = np.float(conc.split('pc')[0])

    if lineshape is None:
        lineshape = best_fit[sensor_id][a].model.name.split('_')[1]
    if lineshape == '1Voigt':
        def _1Voigt_v1_expl(x):
            weightG = best_fit[sensor_id][a].params['weightG'].value
            weightL = best_fit[sensor_id][a].params['weightL'].value
            ampG = best_fit[sensor_id][a].params['ampG'].value
            cenG = best_fit[sensor_id][a].params['cenG'].value
            widG = best_fit[sensor_id][a].params['widG'].value
            ampL = best_fit[sensor_id][a].params['ampL'].value
            cenL = best_fit[sensor_id][a].params['cenL'].value
            widL = best_fit[sensor_id][a].params['widL'].value
            return weightG * (ampG * np.exp(-1 * (np.log(2)) * ((cenG - x) * 2 / widG) ** 2)) + weightL * (
                        ampL / (1 + ((cenL - x) * 2 / widL) ** 2))

        integrand = _1Voigt_v1_expl

    elif lineshape == '1gaussian':
        def _1gaussian_v1_expl(x):
            amp = best_fit[sensor_id][a].params['amp'].value
            cen = best_fit[sensor_id][a].params['cen'].value
            wid = best_fit[sensor_id][a].params['wid'].value
            return amp * np.exp(-1 * (np.log(2)) * ((cen - x) * 2 / wid) ** 2)

        integrand = _1gaussian_v1_expl

    elif lineshape == '1Lorentzian':
        def _1Lorentzian_v1_expl(x):
            amp = best_fit[sensor_id][a].params['amp'].value
            cen = best_fit[sensor_id][a].params['cen'].value
            wid = best_fit[sensor_id][a].params['wid'].value
            return amp / (1 + ((cen - x) * 2 / wid) ** 2)

        integrand = _1Lorentzian_v1_expl

    elif lineshape == '2Voigt_gauss':
        def _2Voigt_gauss_expl(x):
            weightV1 = best_fit[a + ' report'].best_values['weightV1']
            ampG1 = best_fit[a + ' report'].best_values['ampG1']
            cenG1 = best_fit[a + ' report'].best_values['cenG1']
            widG1 = best_fit[a + ' report'].best_values['widG1']
            ampL1 = best_fit[a + ' report'].best_values['ampL1']
            cenL1 = best_fit[a + ' report'].best_values['cenL1']
            widL1 = best_fit[a + ' report'].best_values['widL1']
            weightV2 = best_fit[a + ' report'].best_values['weightV2']
            ampG2 = best_fit[a + ' report'].best_values['ampG2']
            cenG2 = best_fit[a + ' report'].best_values['cenG2']
            widG2 = best_fit[a + ' report'].best_values['widG2']
            ampL2 = best_fit[a + ' report'].best_values['ampL2']
            cenL2 = best_fit[a + ' report'].best_values['cenL2']
            widL2 = best_fit[a + ' report'].best_values['widL2']
            amp3 = best_fit[a + ' report'].best_values['amp3']
            cen3 = best_fit[a + ' report'].best_values['cen3']
            wid3 = best_fit[a + ' report'].best_values['wid3']
            weight1 = best_fit[a + ' report'].best_values['weight1']
            weight2 = best_fit[a + ' report'].best_values['weight2']
            weight3 = best_fit[a + ' report'].best_values['weight3']
            c1 = curve._1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1,
                                  widL=widL1)
            c2 = curve._1Voigt_v2(x, weight=weightV2, ampG=ampG2, cenG=cenG2, widG=widG2, ampL=ampL2, cenL=cenL2,
                                  widL=widL2)
            c3 = curve._1gaussian_v1(x, amp=amp3, cen=cen3, wid=wid3)
            return weight1*c1 + weight2*c2 + weight3*c3

        integrand = _2Voigt_gauss_expl
    elif lineshape == 'Voigt_gauss':
        def _Voigt_gauss_expl(x):
            weightV1 = best_fit[a + ' report'].best_values['weightV1']
            ampG1 = best_fit[a + ' report'].best_values['ampG1']
            cenG1 = best_fit[a + ' report'].best_values['cenG1']
            widG1 = best_fit[a + ' report'].best_values['widG1']
            ampL1 = best_fit[a + ' report'].best_values['ampL1']
            cenL1 = best_fit[a + ' report'].best_values['cenL1']
            widL1 = best_fit[a + ' report'].best_values['widL1']
            amp2 = best_fit[a + ' report'].best_values['amp2']
            cen2 = best_fit[a + ' report'].best_values['cen2']
            wid2 = best_fit[a + ' report'].best_values['wid2']
            weight1 = best_fit[a + ' report'].best_values['weight1']
            weight2 = best_fit[a + ' report'].best_values['weight2']

            c1 = curve._1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1,
                                  widL=widL1)
            c2 = curve._1gaussian_v1(x, amp=amp2, cen=cen2, wid=wid2)
            return weight1 * c1 + weight2 * c2

        integrand = _Voigt_gauss_expl
    else:
        raise ValueError(
            'Chosen lineshape is not defined. Please select either gaussian, lorentzian or voigt lineshape.')

    # integrate - parameters returned
    # y : float; integral of func from `a` to `b`
    # abserr : float; estimate of the absolute error in the result
    integral = integrate.quad(func=integrand, a=limit_integral[0], b=limit_integral[1])

    # combine information in dictionary
    calib_point = dict({'concentration [%]': conc_num, 'integration limit {}'.format(sensor_id): limit_integral,
                        'Integral {}'.format(sensor_id): integral[0],
                        'abs integration error {}'.format(sensor_id): integral[1]})

    return calib_point


def signal_to_noise_ratio(df, sensor, sig_wl=None, noise_wl=None, sw=None):
    if sig_wl is None:
        sig_wl_ = int(df[sensor].apply(pd.to_numeric, errors = 'coerce').idxmax())
        if isinstance(sig_wl_, np.int):
            sig_wl = sig_wl_
        else:
            sig_wl = sig_wl_[0]
        noise_wl_ = int(df[sensor].loc[700:740].apply(pd.to_numeric, errors='coerce').idxmin())
        if isinstance(noise_wl_, np.int):
            noise_wl = noise_wl_
        else:
            noise_wl = noise_wl_[0]
        sw = 5.
    else:
        if noise_wl is None:
            raise ValueError('Wavelength of the background noise is required')
        if sw is None:
            raise ValueError('Slit width is required')
    print('maximal signal at {:.0f}nm; noise calculated at {:.0f}nm'.format(sig_wl, noise_wl))

    # -------------------------------------------------
    signal_av = df[sensor].loc[sig_wl - sw:sig_wl + sw].mean()
    noise = df[sensor].loc[noise_wl - sw:noise_wl + sw].mean()
    snr = (signal_av - noise) / np.sqrt(noise)
    snr_av = snr.mean()

    return snr, snr_av


def sensor_evaluation(l_, sensor_ID, limit_integral, arg, arg_fit, fit_range_pt=None, fit_range_pd=None,
                      data='raw', lineshape=None, info=True, calib_evaluation='maximum', fit_model='simplified',
                      plotting=True):
    if calib_evaluation == 'maximum':
        # multiple files
        calib_point_m = [calibration_point(file_hdr=l_[i][0]['cube']['cube'].filename, info=info,
                                           fitrange_Pt=arg_fit['fit range Pt'], dic_fitting=l_[i][0],
                                           fitrange_Pd=arg_fit['fit range Pd']) for i in range(len(l_))]

        ls_conc = [calib_point_m[c]['concentration [%]'] for c in range(len(calib_point_m))]
        df_calib_point = pd.DataFrame(np.zeros(shape=(len(calib_point_m), 6)), index=ls_conc,
                                      columns=list(calib_point_m[0].keys())[1:]).sort_index()
        for r in range(len(calib_point_m)):
            df_calib_point.loc[calib_point_m[r]['concentration [%]'],
                               'wavelength Pt [nm]'] = calib_point_m[r]['wavelength Pt [nm]']
            df_calib_point.loc[calib_point_m[r]['concentration [%]'],
                               'wavelength Pd [nm]'] = calib_point_m[r]['wavelength Pd [nm]']

            df_calib_point.loc[calib_point_m[r]['concentration [%]'],
                               'intensity Pt [rfu]'] = calib_point_m[r]['intensity Pt [rfu]']
            df_calib_point.loc[calib_point_m[r]['concentration [%]'],
                               'intensity Pd [rfu]'] = calib_point_m[r]['intensity Pd [rfu]']

            df_calib_point.loc[calib_point_m[r]['concentration [%]'], 'std Pt [rfu]'] = calib_point_m[r]['std Pt [rfu]']
            df_calib_point.loc[calib_point_m[r]['concentration [%]'], 'std Pd [rfu]'] = calib_point_m[r]['std Pd [rfu]']

        # label for pre-processing
        id_pt = 'intensity Pt [rfu]'
        id_pt_er = 'std Pt [rfu]'
        id_pd = 'intensity Pd [rfu]'
        id_pd_er = 'std Pd [rfu]'

        # preparation plotting
        label_pt = 'data {:.2f}nm ± {:.2f}nm'.format(df_calib_point['wavelength Pt [nm]'].mean(),
                                                     df_calib_point['wavelength Pt [nm]'].std())
        label_pd = 'data {:.2f}nm ± {:.2f}nm'.format(df_calib_point['wavelength Pd [nm]'].mean(),
                                                     df_calib_point['wavelength Pd [nm]'].std())
    elif calib_evaluation == 'integral':
        if lineshape == '2Voigt_gauss':
            calib_point_int_pt = [integral_calibration_point(file_hdr=l_[i][0]['cube']['cube'].filename,
                                                                   best_fit=l_[i][0]['report MY'], sensor_id='Pt',
                                                                   lineshape=lineshape, limit_integral=limit_integral,
                                                                   a=sensor_ID)
                                  for i in range(len(l_))]
            calib_point_int_pd = [integral_calibration_point(file_hdr=l_[i][0]['cube']['cube'].filename,
                                                                   best_fit=l_[i][0]['report MY'], sensor_id='Pd',
                                                                   limit_integral=limit_integral, lineshape=lineshape,
                                                                   a='Pd-'+sensor_ID.split('-')[1])
                                  for i in range(len(l_))]
        elif lineshape == 'Voigt_gauss':
                calib_point_int_pt = [integral_calibration_point(file_hdr=l_[i][0]['cube']['cube'].filename,
                                                                       best_fit=l_[i][0]['report without ref'],
                                                                       sensor_id='Pt', lineshape=lineshape,
                                                                       limit_integral=limit_integral,
                                                                       a=sensor_ID)
                                      for i in range(len(l_))]
                calib_point_int_pd = [integral_calibration_point(file_hdr=l_[i][0]['cube']['cube'].filename,
                                                                       best_fit=l_[i][0]['report without ref'],
                                                                       sensor_id='Pd', limit_integral=limit_integral,
                                                                       lineshape=lineshape,
                                                                       a='Pd-' + sensor_ID.split('-')[1])
                                      for i in range(len(l_))]
        else:
            calib_point_int_pt = [integral_calibration_point(file_hdr=l_[i][0]['cube']['cube'].filename,
                                                                   best_fit=l_[i][1], sensor_id='Pt',
                                                                   lineshape=lineshape,
                                                                   limit_integral=limit_integral, a=sensor_ID)
                                  for i in range(len(l_))]
            calib_point_int_pd = [integral_calibration_point(file_hdr=l_[i][0]['cube']['cube'].filename,
                                                                   best_fit=l_[i][1], sensor_id='Pd',
                                                                   limit_integral=limit_integral, lineshape=lineshape,
                                                                   a='Pd-' + sensor_ID.split('-')[1])
                                  for i in range(len(l_))]

        ls_conc = [calib_point_int_pt[c]['concentration [%]'] for c in range(len(calib_point_int_pt))]
        df_calib_point = pd.DataFrame(np.zeros(shape=(len(calib_point_int_pt), 6)), index=ls_conc,
                                      columns=list(calib_point_int_pt[0].keys())[1:] + list(
                                          calib_point_int_pd[0].keys())[1:]).sort_index()

        for r in range(len(calib_point_int_pt)):
            df_calib_point.loc[calib_point_int_pt[r]['concentration [%]'],
                               'integration limit Pt'] = str(calib_point_int_pt[r]['integration limit Pt'])
            df_calib_point.loc[calib_point_int_pd[r]['concentration [%]'],
                               'integration limit Pd'] = str(calib_point_int_pd[r]['integration limit Pd'])

            df_calib_point.loc[calib_point_int_pt[r]['concentration [%]'], 'Integral Pt'] = calib_point_int_pt[r][
                'Integral Pt']
            df_calib_point.loc[calib_point_int_pd[r]['concentration [%]'], 'Integral Pd'] = calib_point_int_pd[r][
                'Integral Pd']

            df_calib_point.loc[calib_point_int_pt[r]['concentration [%]'],
                               'abs integration error Pt'] = calib_point_int_pt[r]['abs integration error Pt']
            df_calib_point.loc[calib_point_int_pd[r]['concentration [%]'],
                               'abs integration error Pd'] = calib_point_int_pd[r]['abs integration error Pd']

        # label for pre-processing
        id_pt = 'Integral Pt'
        id_pt_er = 'abs integration error Pt'
        id_pd = 'Integral Pd'
        id_pd_er = 'abs integration error Pd'

        # preparation plotting
        label_pt = 'data'
        label_pd = 'data'
    else:
        raise ValueError('Chose either "maximum" or "integral" for evaluation of the calibration points.')

    # ===================================================================================================
    # data pre-processing
    if data == 'raw':
        # i0 / i
        df_pt_ = df_calib_point[id_pt]
        df_pt = df_pt_.loc[0] / df_pt_
        if calib_evaluation == 'maximum':
            error_pt = df_calib_point[id_pt_er] * 2
        else:
            error_pt = df_calib_point[id_pt_er]
        df_pd_ = df_calib_point[id_pd]
        df_pd = df_pd_.loc[0] / df_pd_
        if calib_evaluation == 'maximum':
            error_pd = df_calib_point[id_pd_er] * 2
        else:
            error_pd = df_calib_point[id_pd_er]

    elif data == 'i0/i':
        if len(df_calib_point[id_pt].loc[0.0]) > 1:
            dfPt_0pc = df_calib_point[id_pt].loc[0.0].mean()
            dfPd_0pc = df_calib_point[id_pd].loc[0.0].mean()
        else:
            dfPt_0pc = df_calib_point[id_pt].loc[0.0]
            dfPd_0pc = df_calib_point[id_pd].loc[0.0]
        df_pt = dfPt_0pc / df_calib_point[id_pt]
        error_pt = []
        df_pd = dfPd_0pc / df_calib_point[id_pd]
        error_pd = []

    elif data == 'normalized':
        print('in progress')
        error_pt = []
        error_pd = []
    else:
        raise ValueError('decide on the pre-processing of the data. choose either raw, i07i or normalized')

    df_pt = df_pt.drop_duplicates()
    df_pd = df_pd.drop_duplicates()
    yfit_pt = df_pt.loc[fit_range_pt[0]:fit_range_pt[1]].values
    xfit_pt = df_pt.loc[fit_range_pt[0]:fit_range_pt[1]].index
    yfit_pd = df_pd.loc[fit_range_pd[0]:fit_range_pd[1]].values
    xfit_pd = df_pd.loc[fit_range_pd[0]:fit_range_pd[1]].index

    # ===================================================================================================
    # curve fitting tsm Stern-Volmer
    # f, ksv values from https://pubs.acs.org/doi/10.1021/ac801521v -- k_vs at 25C
    if fit_model == 'tsm':
        print('Two-site model for Stern-Volmer Fit')
        tsm_sv = Model(curve._tsm_sternvolmer)  #
        params_pt = tsm_sv.make_params(f=0.87, m=0.4, k=0.165)
        params_pd = tsm_sv.make_params(f=0.87, m=0.4, k=0.92)

        params_pt['f'].min = 0.
        params_pt['f'].max = 1.
        params_pd['f'].min = 0.
        params_pd['f'].max = 1.
        params_pt['m'].min = 0.
        params_pd['m'].min = 0.

        # i/i0 Fit
        result_pt = tsm_sv.fit(1 / yfit_pt, params_pt, x=xfit_pt, nan_policy='omit')
        result_pd = tsm_sv.fit(1 / yfit_pd, params_pd, x=xfit_pd, nan_policy='omit')
    else:
        print('Simplified model for Stern-Volmer Fit')
        simply_sv = Model(curve. _sternvolmer)
        params_pt = simply_sv.make_params(k=0.165, f=0.87)
        params_pd = simply_sv.make_params(k=0.92, f=0.87)

        params_pt['k'].min = 0.
        params_pd['k'].min = 0.
        params_pt['f'].min = 0.
        params_pt['f'].max = 1.
        params_pd['f'].min = 0.
        params_pd['f'].max = 1.

        # i0/i Fit
        result_pt = simply_sv.fit(yfit_pt, params_pt, x=xfit_pt, nan_policy='omit')
        result_pd = simply_sv.fit(yfit_pd, params_pd, x=xfit_pd, nan_policy='omit')

    # ===================================================================================================
    # data post-processing
    if data == 'raw':
        df_pt_out = df_pt_
        df_pd_out = df_pd_
    else:
        # 'i0/i' and 'normalized
        df_pt_out = df_pt
        df_pd_out = df_pd

    # ===================================================================================================
    if plotting is True:
        [fig,
         ax] = plot.plotting_2individual_sensor_evaluated(df_pt=df_pt, error_pt=error_pt, label_pt=label_pt,  arg=arg,
                                                          xfit_pt=xfit_pt, yfit_pt=yfit_pt, result_pt=result_pt,
                                                          df_pd=df_pd, error_pd=error_pd, label_pd=label_pd,
                                                          xfit_pd=xfit_pd, yfit_pd=yfit_pd, result_pd=result_pd,
                                                          fontsize_=arg['fontsize'], fit_model=fit_model)
        plt.show()
    else:
        plt.ioff()
        [fig,
         ax] = plot.plotting_2individual_sensor_evaluated(df_pt=df_pt, error_pt=error_pt, label_pt=label_pt, arg=arg,
                                                          xfit_pt=xfit_pt, yfit_pt=yfit_pt, result_pt=result_pt,
                                                          df_pd=df_pd, error_pd=error_pd, label_pd=label_pd,
                                                          xfit_pd=xfit_pd, yfit_pd=yfit_pd, result_pd=result_pd,
                                                          fontsize_=arg['fontsize'], fit_model=fit_model)
        plt.close(fig)

    # ===================================================================================================
    # combine information for output
    sensor_para = dict({'Pt - tsmFit': result_pt, 'Pt - dataFit': df_pt_out, 'Pt - data raw': df_calib_point[id_pt],
                        'Pt - error': error_pt, 'Pd - tsmFit': result_pd, 'Pd - dataFit': df_pd_out,
                        'Pd - data raw': df_calib_point[id_pd], 'Pd - error': error_pd})

    return df_calib_point, sensor_para, fig, ax


def integral_sensor(l_dual, ls_conc, lambda_dual, sensorID):
    # -----------------------------------------------
    # define integrand
    def _2Voigt_gauss_expl(x):
        weightV1 = bestFit['weightV1']
        ampG1 = bestFit['ampG1']
        cenG1 = bestFit['cenG1']
        widG1 = bestFit['widG1']
        ampL1 = bestFit['ampL1']
        cenL1 = bestFit['cenL1']
        widL1 = bestFit['widL1']
        weightV2 = bestFit['weightV2']
        ampG2 = bestFit['ampG2']
        cenG2 = bestFit['cenG2']
        widG2 = bestFit['widG2']
        ampL2 = bestFit['ampL2']
        cenL2 = bestFit['cenL2']
        widL2 = bestFit['widL2']
        amp3 = bestFit['amp3']
        cen3 = bestFit['cen3']
        wid3 = bestFit['wid3']
        weight1 = bestFit['weight1']
        weight2 = bestFit['weight2']
        weight3 = bestFit['weight3']

        c1 = curve._1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1,
                              widL=widL1)
        c2 = curve._1Voigt_v2(x, weight=weightV2, ampG=ampG2, cenG=cenG2, widG=widG2, ampL=ampL2, cenL=cenL2,
                              widL=widL2)
        c3 = curve._1gaussian_v1(x, amp=amp3, cen=cen3, wid=wid3)
        return weight1 * c1 + weight2 * c2 + weight3 * c3

    def _Voigt_gauss_expl(x):
        weightV1 = bestFit['weightV1']
        ampG1 = bestFit['ampG1']
        cenG1 = bestFit['cenG1']
        widG1 = bestFit['widG1']
        ampL1 = bestFit['ampL1']
        cenL1 = bestFit['cenL1']
        widL1 = bestFit['widL1']
        amp2 = bestFit['amp2']
        cen2 = bestFit['cen2']
        wid2 = bestFit['wid2']
        weight1 = bestFit['weight1']
        weight2 = bestFit['weight2']

        c1 = curve._1Voigt_v2(x, weight=weightV1, ampG=ampG1, cenG=cenG1, widG=widG1, ampL=ampL1, cenL=cenL1,
                              widL=widL1)
        c2 = curve._1gaussian_v1(x, amp=amp2, cen=cen2, wid=wid2)
        return weight1 * c1 + weight2 * c2

    # -----------------------------------------------
    dual_int = pd.DataFrame(np.zeros(shape=(0, 2)), columns=['Integral', 'abs error'])

    for i in range(len(ls_conc)):
        bestFit = l_dual[i]['fit sensor'][sensorID].best_values
        if '+' in sensorID:
            integrand = _2Voigt_gauss_expl
        else:
            integrand = _Voigt_gauss_expl

        conc = l_dual[i]['concentration']
        integral = integrate.quad(func=integrand, a=lambda_dual[0], b=lambda_dual[1])
        dual_int.loc[conc] = integral
    dual_int = dual_int.sort_index()

    # -----------------------------------------------------------------
    # simplified TSM fit for (normalized) pre-factors
    simply_sv = Model(curve._sternvolmer)
    params_dual_int = simply_sv.make_params(k=0.165 * 2, f=0.87 * 2)

    params_dual_int['k'].min = 0.
    params_dual_int['f'].min = 0.
    params_dual_int['f'].max = 1.

    # use i0/i data for fit and re-calculate i afterwards
    # full concentration range
    ytofit_dual_int = dual_int.loc[0, 'Integral'] / dual_int['Integral'].values
    xtofit_dual_int = dual_int.index
    result_intDual = simply_sv.fit(ytofit_dual_int, params_dual_int, x=xtofit_dual_int, nan_policy='omit')

    df_bestFit_maxDual = pd.DataFrame(dual_int.loc[0, 'Integral'] / result_intDual.best_fit, index=xtofit_dual_int)
    dual_int['SV Fit'] = df_bestFit_maxDual

    return dual_int
