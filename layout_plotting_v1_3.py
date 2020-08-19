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
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.widgets import RectangleSelector
from spectral import *
from scipy import ndimage
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import seaborn as sns
import random
import math

import correction_hyperCamera_v1_4 as corr

sns.set(style="ticks")


# =====================================================================================
class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.0f}, y={:.0f}, z={:.03f}'.format(x, y, z)


def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    print('startposition: ({:.0f}, {:.0f})'.format(eclick.xdata, eclick.ydata))
    print('endposition  : ({:.0f}, {:.0f})'.format(erelease.xdata, erelease.ydata))
    print('used button  : ', eclick.button)


def toggle_selector(event):
    print('Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


# =====================================================================================
def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB])


def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''

    return {"hex": [RGB_to_hex(RGB) for RGB in gradient],
            "r": [RGB[0] for RGB in gradient],
            "g": [RGB[1] for RGB in gradient],
            "b": [RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''

    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)

    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j]))
            for j in range(3)]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)


# ============================================================
def plotting_spectra(xlabel, ylabel, ylabel2=None, fontsize_=12, figsize_=(5., 4.5)):
    fig, ax = plt.subplots(figsize=figsize_)
    ax2 = ax.twinx()
    ax2y = ax2.twiny()

    ax.set_xlabel(xlabel, fontsize=fontsize_, color='navy')
    ax.set_ylabel(ylabel, fontsize=fontsize_, color='crimson')
    if ylabel2 is None:
        pass
    else:
        ax2.set_ylabel(ylabel2, fontsize=fontsize_)

    ax.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_*0.9)
    if ylabel2 is None:
        ax2.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_*0)
    else:
        ax2.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_*0.9)
    ax2y.tick_params(axis='both', which='both', direction='in', labelsize=0)

    return fig, ax, ax2, ax2y


def plotting_3optodes(arg, df_sensor1, df_sensor2, df_sensor3):
    """
    Common plot of sensors - sensor3 is the superimposed signal of sensor1 and sensor2. Provide a list of the sensor
    names
    :param arg:
    :param df_sensor1:
    :param df_sensor2:
    :param df_sensor3:
    :return:
    """
    # load arguments
    if 'label' in arg.keys():
        ls_label = arg['label']
    else:
        ls_label = ['sensor1', 'sensor2', 'superimposed signal']
    if 'figure size meas' in arg.keys():
        figsize = arg['figure size meas']
    else:
        figsize=(5., 4.5)
    if 'fontsize meas' in arg.keys():
        fontsize_ = arg['fontsize meas']
    else:
        fontsize_ = 13
    if 'xlabel' in arg.keys():
        xlabel_ = arg['xlabel']
    else:
        xlabel_ = 'Wavelength [nm]'
    if 'ylabel' in arg.keys():
        ylabel_ = arg['ylabel']
    else:
        ylabel_ = 'Rel. intensity [rfu]'

    # ==========================================================================
    fig, ax = plt.subplots(figsize=figsize)
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    # Dual indicator
    check = 'Du'
    for idx in ls_label:
        if idx[:2].lower() == check.lower():
            dlabel = idx
    ax.plot(df_sensor3, lw=1.5, color='k', label=dlabel)

    # Single indicator
    check = 'Pt'
    check2 = 'Pd'
    for idx in ls_label:
        if idx[:2].lower() == check.lower():
            pt_label = idx
        elif idx[:2].lower() == check2.lower():
            pd_label = idx
    ax.plot(df_sensor1, lw=1.25, ls='--', color='#FF8C00', label=pt_label)
    ax.plot(df_sensor2, lw=1.25, ls='--', color='#077b8a', label=pd_label)

    # legend
    ax.legend(frameon=True, fancybox=True, loc=0, fontsize=fontsize_ * 0.7)

    # layout for axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    axT.xaxis.set_major_locator(ticker.MultipleLocator(50))
    axT.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    axR.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    axR.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    ax.tick_params(axis='both', which='both', direction='out', labelsize=fontsize_ * 0.8)
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)

    ax.set_xlabel(xlabel_, fontsize=fontsize_)
    ax.set_ylabel(ylabel_, fontsize=fontsize_)

    return fig, ax


def plotting_dualoptodes(df_dual, arg):

    fig, ax = plt.subplots(figsize=arg['figure size'])
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    for en, c in enumerate(df_dual.keys()):
        ax.plot(df_dual[c], lw=1.5, color=arg['color dual ' + str(en+1)], label=c)

    # legend
    ax.legend(frameon=True, fancybox=True, loc=0, fontsize=arg['fontsize']*0.7)

    # layout for axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    axT.xaxis.set_major_locator(ticker.MultipleLocator(50))
    axT.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    ax.set_xlabel('Wavelength [nm]', fontsize=arg['fontsize'])
    ax.set_ylabel('Intensity [rfu]', fontsize=arg['fontsize'])

    ax.tick_params(axis='both', which='both', direction='in', labelsize=arg['fontsize'] * 0.9)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)

    return fig, ax


def plotting_fit_results_2Sensors(df_sensors, df_sensors_std, arg, arg_fit):
    fig, ax = plt.subplots(figsize=arg['figure size meas'])
    plt.title('Generic function single indicator', x=0.3, y=1.08)
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    for en, i in enumerate(df_sensors.columns):
        if 'color ' + i.split('-')[0] in arg:
            color_ = arg['color ' + i.split('-')[0]]
        else:
            color_ = colors[en]
        ax.plot(df_sensors[i], lw=0.75, label=i)
        ref = arg_fit['fit range ref'][1]
        ax.fill_between(df_sensors.loc[ref:].index, df_sensors[i].loc[ref:] - df_sensors_std.loc[ref:][i],
                        df_sensors.loc[ref:][i] + df_sensors_std.loc[ref:][i], color='grey', alpha=0.5)

    ax.legend(frameon=True, fancybox=True, fontsize=arg['fontsize meas'] * 0.7)

    ax.tick_params(axis='both', which='both', direction='out', labelsize=arg['fontsize meas'] * 0.9)
    axR.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axT.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    ax.set_xlabel('Wavelength [nm]', fontsize=arg['fontsize meas'])
    ax.set_ylabel('Norm. Intensity', fontsize=arg['fontsize meas'])
    plt.tight_layout()
    plt.show()

    return fig, ax


def plotting_singleFit_maximal(dic_SVFit, df_calib, singleID, arg, simply, unit, analyte, ratiometric):
    color = [arg['color ' + s.split('-')[0]] for s in singleID]

    fig, ax = plt.subplots(figsize=arg['figure size meas'])
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    for en, s in enumerate(singleID):
        # calibration points and fit
        ax.plot(1 / dic_SVFit['data norm averaged'][s.split(' ')[0] + ' mean'], color=color[en], lw=0, ms=4, marker='o',
                fillstyle='none', label=s)

        Fitnorm = dic_SVFit[s + ' Fit']
        Reportav = dic_SVFit[s + ' Fit']['Report']

        if simply is True:
            ax.plot(1 / Fitnorm['norm best Fit'], lw=1., color=color[en], ls='-.',
                    label='Fit $χ^2$={:.2e} \n f={:.3f}, K$_{}$$_{}$={:.3f}'.format(Reportav.redchi,
                                                                                    Reportav.best_values['f'], 'S', 'V',
                                                                                    Reportav.best_values['k']))
        else:
            ax.plot(1 / Fitnorm['norm best Fit'], lw=1., color=color[en], ls='-.',
                    label='Fit $χ^2$={:.2e} \n f={:.3f}, K$_{}$$_{}$$_{}$={:.3f}, '
                          'K$_{}$$_{}$$_{}$={:.3f}'.format(Reportav.redchi, Reportav.best_values['f'], 'S', 'V', '1',
                                                           Reportav.best_values['k'], 'S', 'V', '2',
                                                           Reportav.best_values['k']*Reportav.best_values['m']))
    ax.legend(fontsize=arg['fontsize meas'] * 0.7)

    # adding error bars
    for en, s in enumerate(singleID):
        er = dic_SVFit['data norm averaged'][s.split(' ')[0] + ' mean']
        d = df_calib.filter(like=s).filter(like='STD').mean(axis=1)
        yerr = np.abs((d - d.loc[0]) / d.loc[0])
        ax.errorbar(x=er.index, y=1 / er, yerr=yerr, fmt='o', ms=4, mfc='white', color=color[en])

    if unit == '%air':
        unit_ = '(%air)'
    elif unit == 'hPa':
        unit_ = '[hPa]'
    else:
        print('Unit undknown - define for plotting in line 1152')

    ax.set_xlabel('Concentration ' + analyte + ' ' + unit_, fontsize=arg['fontsize meas'])
    if ratiometric is True:
        ax.set_ylabel('Intensity ratio $R_0$/R ', fontsize=arg['fontsize meas'])
    else:
        ax.set_ylabel('Intensity ratio $I_0$/I ', fontsize=arg['fontsize meas'])
    ax.tick_params(axis='both', which='both', direction='out', labelsize=arg['fontsize meas'] * 0.9)
    axR.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axT.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    plt.tight_layout()

    return fig, ax


def plotting_singleFit_integral(dic_SVFit, df_data, df_norm_av, singleID, arg, simply, unit, analyte, ratiometric):
    color = [arg['color ' + s.split('-')[0]] for s in singleID]

    fig, ax = plt.subplots(figsize=arg['figure size meas'])
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    for en, s in enumerate(singleID):
        # calibration points and fit
        ax.plot(1 / dic_SVFit['data norm averaged'][s.split(' ')[0] + ' mean'], color=color[en], lw=0, marker='o', ms=3,
                fillstyle='none', label=s)
        Fitnorm = dic_SVFit[s + ' Fit']
        Reportav = dic_SVFit[s + ' Fit']['Report']
        if simply is True:
            ax.plot(1 / Fitnorm['norm best Fit'], lw=1., color=color[en], ls='-.',
                label='Fit $χ^2$={:.2e} \n f={:.3f}, K$_{}$$_{}$={:.3f}'.format(Reportav.redchi,
                                                                                Reportav.best_values['f'], 'S', 'V',
                                                                                Reportav.best_values['k']))
        else:
            ax.plot(1 / Fitnorm['norm best Fit'], lw=1., color=color[en], ls='-.',
                    label='Fit $χ^2$={:.2e} \n f={:.3f}, K$_{}$$_{}$$_{}$={:.3f}, '
                          'K$_{}$$_{}$$_{}$={:.3f}'.format(Reportav.redchi, Reportav.best_values['f'], 'S', 'V', '1',
                                                           Reportav.best_values['k'], 'S', 'V', '2',
                                                           Reportav.best_values['k']*Reportav.best_values['m']))

    ax.legend(fontsize=arg['fontsize meas'] * 0.7)

    # adding error bars
    for en, s in enumerate(singleID):
        d = df_data[s + ' STD']
        yerr = np.abs((d - d.loc[0]) / d.loc[0])
        er = dic_SVFit['data norm averaged'][s.split(' ')[0] + ' mean']
        ax.errorbar(x=df_norm_av.index, y=1 / er, yerr=yerr, fmt='o', ms=4, mfc='white', color=color[en])

    if unit == '%air':
        unit_ = '(%air)'
    elif unit == 'hPa':
        unit_ = '[hPa]'
    else:
        print('Unit unknown - define for plotting in line 1152')

    ax.set_xlabel('Concentration ' + analyte + ' ' + unit_, fontsize=arg['fontsize meas'])
    if ratiometric is True:
        ax.set_ylabel('Intensity ratio $R_0$/R ', fontsize=arg['fontsize meas'])
    else:
        ax.set_ylabel('Intensity ratio $I_0$/I ', fontsize=arg['fontsize meas'])
    ax.tick_params(axis='both', which='both', direction='out', labelsize=arg['fontsize meas'] * 0.9)
    axR.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axT.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    plt.tight_layout()

    return fig, ax


def plotting_fitresults(xdata, ddf, result, fit, arg, col_data='darkorange', plot_data=True, fig=None, ax=None,
                        ax_dev=None, color_fit='k'):
    if ax is None:
        fig, (ax, ax_dev) = plt.subplots(figsize=arg['figure size meas'], nrows=2)

    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    ax_dev.get_shared_x_axes().join(ax_dev, ax)
    axR_dev = ax_dev.twiny()
    axT_dev = ax_dev.twinx()
    axR_dev.get_shared_x_axes().join(axR_dev, ax_dev)
    axT_dev.get_shared_x_axes().join(axT_dev, ax_dev)

    # -------------------------------------------------
    # plotting
    if plot_data is True:
        ax.plot(ddf, color=col_data, lw=0., marker='o', fillstyle='none', label='data')

    # reduced chi-square for goodness of fit ~ standard error of the regression -> small as possible
    ax.plot(xdata, result.best_fit, 'k--', label='{} fit χ2 = {:.2e}'.format(fit, result.redchi))
    ax_dev.plot(xdata, result.residual, lw=0., marker='.', color=color_fit)
    ax_dev.axhline(0, color='k', lw=0.5)
    ax.legend(loc=2, fontsize=arg['fontsize Fit']*0.8)

    # -------------------------------------------------
    # layout axes
    if result.best_fit.max() > ddf.max():
        ymax = result.best_fit.max()*1.15
    else:
        ymax = ddf.max()*1.15
    if int(ymax) == 0:
        if (int(ymax * 10)) == 0:
            if (int(ymax * 100)) == 0:
                ymin = -(1 / 10000)
            else:
                ymin = -(1 / 1000)
        else:
            ymin = -(1 / 100)
    else:
        ymin = -(1 / 10)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', which='both', direction='out', labelsize=arg['fontsize Fit']*.9)
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)
    ax_dev.tick_params(axis='both', which='both', direction='out', labelsize=arg['fontsize Fit']*.9)
    axR_dev.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT_dev.tick_params(axis='both', which='both', direction='in', labelsize=0)

    # label axes
    ax_dev.set_xlabel('Wavelength [nm]', fontsize=arg['fontsize Fit'])
    ax.set_ylabel('Norm. intensity [a.u.]', fontsize=arg['fontsize Fit'])
    ax_dev.set_ylabel('Residuals', fontsize=arg['fontsize Fit'])
    plt.subplots_adjust(hspace=.15)

    return fig, ax, ax_dev


def plotting_fitresults_dual(df_dual, df_tofit_dual, result_dual, title):
    fig_dual = plt.figure()
    ax_dual = plt.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
    axR_dual = ax_dual.twinx()
    axT_dual = ax_dual.twiny()
    axR_dual.get_shared_y_axes().join(axR_dual, ax_dual)
    axT_dual.get_shared_x_axes().join(axT_dual, ax_dual)

    ax_dual.set_title(title)

    ax_dual_dev = plt.subplot2grid((3, 1), (2, 0), colspan=2)
    ax_dual_dev.get_shared_x_axes().join(ax_dual_dev, ax_dual)
    axR_dual_dev = ax_dual_dev.twiny()
    axT_dual_dev = ax_dual_dev.twinx()
    axR_dual_dev.get_shared_x_axes().join(axR_dual_dev, ax_dual_dev)
    axT_dual_dev.get_shared_x_axes().join(axT_dual_dev, ax_dual_dev)

    # -------------------------------------------------
    # plotting
    ax_dual.plot(df_dual, color='#7fa998', lw=0., marker='.', label='Pd/Pt-TPTBP data')
    # reduced chi-quare for goodness of fit
    ax_dual.plot(df_tofit_dual.index, result_dual.best_fit, 'k--', label='best fit χ2 = {:.2e}'.format(result_dual.redchi))
    ax_dual_dev.plot(df_tofit_dual.index, result_dual.residual, lw=0., marker='.', color='k')
    ax_dual_dev.axhline(0, color='k', lw=0.5)
    ax_dual.legend()

    # -------------------------------------------------
    # layout axes
    # ax_dual.set_ylim(-0.005, 0.03)
    y_lim = round(np.abs(result_dual.residual.min()) * 1.05, 2)
    ax_dual_dev.set_ylim(-y_lim, y_lim)
    ax_dual.tick_params(axis='both', which='both', direction='out')
    axR_dual.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT_dual.tick_params(axis='both', which='both', direction='in', labelsize=0)
    ax_dual_dev.tick_params(axis='both', which='both', direction='out')
    axR_dual_dev.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT_dual_dev.tick_params(axis='both', which='both', direction='in', labelsize=0)

    # label axes
    ax_dual_dev.set_xlabel('Wavelength [nm]')
    ax_dual.set_ylabel('Norm. intensity [a.u.]')
    ax_dual_dev.set_ylabel('Residuals')

    plt.tight_layout(h_pad=-1.5)

    return fig_dual, ax_dual


def plotting_2individual_sensor_evaluated(df_pt, error_pt, label_pt, xfit_pt, yfit_pt, result_pt, df_pd, error_pd,
                                          label_pd, xfit_pd, yfit_pd, result_pd, arg, fit_model, fig=None, ax=None,
                                          fontsize_=12):
    if 'color Pt' not in arg.keys():
        raise ValueError('color for individual sensors required')

    if fig is None:
        fig_calib, (ax_pt, ax_pd) = plt.subplots(figsize=(9, 4), ncols=2)
        if fit_model == ' tsm':
            fig_calib.suptitle('two-site model Stern-Volmer Fit', fontsize=fontsize_ * 1.1)
        else:
            fig_calib.suptitle('simplified Stern-Volmer Fit', fontsize=fontsize_ * 1.1)
    else:
        fig_calib = fig
        ax_pt = ax[0]
        ax_pd = ax[1]
    axR = ax_pt.twinx()
    axT = ax_pt.twiny()
    axR1 = ax_pd.twinx()
    axT1 = ax_pd.twiny()
    axR.get_shared_y_axes().join(axR, ax_pt)
    axT.get_shared_x_axes().join(axT, ax_pt)
    axR1.get_shared_y_axes().join(axR1, ax_pd)
    axT1.get_shared_x_axes().join(axT1, ax_pd)

    # -----------------------------------------------------------------------------------------
    # Pt-TPTBP sensor 1
    ax_pt.plot(df_pt, lw=0., marker='o', markeredgewidth=1.5, fillstyle='none', color=arg['color Pt'],
               label=label_pt)
    if isinstance(error_pt, list):
        pass
    else:
        ax_pt.errorbar(x=xfit_pt, y=yfit_pt, yerr=error_pt, fmt='none', lw=0.75, ecolor='k')

    if fit_model == 'tsm':
        ax_pt.plot(xfit_pt, 1 / result_pt.best_fit, lw=1.5, ls='--', color=arg['color tsm'],
                   label='fit $Χ^2$={:.2e}'.format(result_pt.redchi))
    else:
        ax_pt.plot(xfit_pt, result_pt.best_fit, lw=1.5, ls='--', color=arg['color tsm'],
                   label='fit $Χ^2$={:.2e}'.format(result_pt.redchi))

    # -----------------------------------------------------
    # Pd-TPTBP sensor 2
    ax_pd.plot(df_pd, lw=0., marker='o', markeredgewidth=1.5, fillstyle='none', color=arg['color Pd'],
               label=label_pd)

    if isinstance(error_pd, list):
        pass
    else:
        ax_pd.errorbar(x=xfit_pd, y=yfit_pd, yerr=error_pd, fmt='none', lw=0.75, ecolor='k')
    if fit_model == 'tsm':
        ax_pd.plot(xfit_pd, 1 / result_pd.best_fit, lw=1.5, ls='--', color=arg['color tsm'],
                   label='fit $Χ^2$={:.2e}'.format(result_pd.redchi))
    else:
        ax_pd.plot(xfit_pd, result_pd.best_fit, lw=1.5, ls='--', color=arg['color tsm'],
                   label='fit $Χ^2$={:.2e}'.format(result_pd.redchi))

    # ------------------------------------
    ax_pt.legend(loc=0, frameon=True, fancybox=True, fontsize=fontsize_ * .8)
    ax_pd.legend(loc=0, frameon=True, fancybox=True, fontsize=fontsize_ * .8)

    # -----------------------------------------------------------------------------------------
    ax_pt.tick_params(axis='both', which='both', direction='out', labelsize=fontsize_ * 0.9)
    ax_pd.tick_params(axis='both', which='both', direction='out', labelsize=fontsize_ * 0.9)
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axR1.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT1.tick_params(axis='both', which='both', direction='in', labelsize=0)

    ax_pt.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_pt.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    axT.xaxis.set_major_locator(ticker.MultipleLocator(20))
    axT.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax_pd.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_pd.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    axT1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    axT1.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    # -----------------------------------------------------------------------------------------
    # axes label
    ax_pt.set_xlabel('Concentration [%]', fontsize=fontsize_)
    ax_pd.set_xlabel('Concentration [%]', fontsize=fontsize_)
    ax_pt.set_ylabel('$I_0/I$ Pt', fontsize=fontsize_)
    ax_pd.set_ylabel('$I_0/I$ Pd', fontsize=fontsize_)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig_calib, (ax_pt, ax_pd)


def plotting_averagedSignal(cube_corr, figsize_, conc=None, analyte=None, unit=None, fontsize_=13.,
                           plot_range=(500, 900)):
    fig, ax = plt.subplots(figsize=figsize_)
    if conc is None or analyte is None or unit is None:
        plt.title('Averaged Sensor Signal of RoI', y=1.08)
    else:
        plt.title('Averaged Sensor Signal at ' + conc + ' ' + analyte + ' (' + unit + ')', y=1.08)
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    n = len(cube_corr['average data'].filter(like='mean', axis=1).columns)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for en, i in enumerate(cube_corr['average data'].filter(like='mean', axis=1).columns.tolist()):
        ax.plot(cube_corr['average data'].loc[plot_range[0]:plot_range[1],:].filter(like='mean', axis=1)[i], marker='.',
                lw=0, color=colors[en], label=i.split('mean')[0])
        j = i.split('mean')[0] + 'STD'

        mean = cube_corr['average data'].loc[plot_range[0]:plot_range[1],:].filter(like='mean', axis=1)[i]
        sigma = cube_corr['average data'].loc[plot_range[0]:plot_range[1],:].filter(like='STD', axis=1)[j]
        ax.fill_between(cube_corr['average data'].loc[plot_range[0]:plot_range[1],:].filter(like='mean', axis=1).index, mean + sigma, mean - sigma,
                        facecolor='grey', alpha=0.3)
    ax.set_xlim(plot_range[0], plot_range[1])
    ax.legend(frameon=True, fancybox=True, fontsize=fontsize_*0.7)
    ax.tick_params(axis='both', which='both', direction='out', labelsize=fontsize_ * 0.9)
    axR.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_* 0.9, labelcolor='white')
    axT.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_* 0.9, labelcolor='white')
    ax.set_xlabel('Wavelength [nm]', fontsize=fontsize_)
    ax.set_ylabel('Rel. Intensity', fontsize=fontsize_)
    plt.tight_layout()
    plt.show()

    return fig, ax

def validation_example_pixel(name, dic_sens, res_sensor1, res_sensor2, df_sig1_crop, df_sig2_crop, res_bestFit, arg):
    # Validation
    pw_ex = random.choice(res_sensor1.index)
    ph_ex = random.choice(res_sensor1.columns)

    # --------------------------------------------------------------------------------------------------
    # plotting
    fig_val = plt.figure(figsize=(5, 5))
    fig_val.suptitle('LC of {} at {}x{}'.format(name, pw_ex, ph_ex), fontsize=10.5, x=0.37)

    ax_val = plt.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
    axR = ax_val.twinx()
    axT = ax_val.twiny()
    axR.get_shared_y_axes().join(axR, ax_val)
    axT.get_shared_x_axes().join(axT, ax_val)

    ax_dev = plt.subplot2grid((3, 1), (2, 0), colspan=2)
    ax_dev.get_shared_x_axes().join(ax_dev, ax_val)
    axR1 = ax_dev.twiny()
    axT1 = ax_dev.twinx()
    axR1.get_shared_x_axes().join(axR1, ax_dev)
    axT1.get_shared_x_axes().join(axT1, ax_dev)

    # ------------------------------------------------------------------
    # plot curves
    df_sum = res_sensor1.loc[pw_ex, ph_ex] * df_sig1_crop + res_sensor2.loc[pw_ex, ph_ex] * df_sig2_crop

    ax_val.plot(df_sum, ls='--', color='k', label='$LC$ $S$ ~{:.2e}'.format(res_bestFit.loc[pw_ex, ph_ex]))
    ax_val.plot(dic_sens[pw_ex][ph_ex], lw=0, marker='.', fillstyle='none', color=arg['color dual 1'], label='data')
    ax_val.plot(res_sensor1.loc[pw_ex, ph_ex] * df_sig1_crop, lw=0.75, color=arg['color Pt'],
                label='α {:.2e}'.format(res_sensor1.loc[pw_ex, ph_ex]))
    ax_val.plot(res_sensor2.loc[pw_ex, ph_ex] * df_sig2_crop, lw=0.75, color=arg['color Pd'],
                label='α {:.2e}'.format(res_sensor2.loc[pw_ex, ph_ex]))
    ax_val.legend(fontsize=9, frameon=True, fancybox=True)

    # plot residuals
    ax_dev.plot(df_sum - dic_sens[pw_ex][ph_ex], lw=0, marker='.', fillstyle='none', color=arg['color dual 1'])
    ax_dev.axhline(0, color='k', lw=0.5)

    # ------------------------------------------------------------------
    # layout axes
    ax_val.tick_params(axis='both', which='both', direction='out')
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)
    ax_dev.tick_params(axis='both', which='both', direction='out')
    axR1.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT1.tick_params(axis='both', which='both', direction='in', labelsize=0)

    # label axes
    ax_dev.set_xlabel('Wavelength [nm]')
    ax_val.set_ylabel('Norm. intensity [rfu]')
    ax_dev.set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()

    return fig_val


def validation_example_calib(name, dic_sens, res_sensor1, res_sensor2, df_sig1_crop, df_sig2_crop, res_bestFit, arg):
    # Validation
    c = random.choice([c for c in dic_sens.keys()])

    # --------------------------------------------------------------------------------------------------
    # plotting
    fig_val = plt.figure(figsize=(5, 5))
    fig_val.suptitle('LC of {} at {}% $O_2$'.format(name, c), fontsize=10, x=0.37)

    ax_val = plt.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
    axR = ax_val.twinx()
    axT = ax_val.twiny()
    axR.get_shared_y_axes().join(axR, ax_val)
    axT.get_shared_x_axes().join(axT, ax_val)

    ax_dev = plt.subplot2grid((3, 1), (2, 0), colspan=2)
    ax_dev.get_shared_x_axes().join(ax_dev, ax_val)
    axR1 = ax_dev.twiny()
    axT1 = ax_dev.twinx()
    axR1.get_shared_x_axes().join(axR1, ax_dev)
    axT1.get_shared_x_axes().join(axT1, ax_dev)

    # ------------------------------------------------------------------
    # plot curves
    df_sum = res_sensor1.loc[c, 0] * df_sig1_crop + res_sensor2.loc[c, 0] * df_sig2_crop

    ax_val.plot(df_sum, ls='--', color='k', label='$LC$ $S$ ~{:.2e}'.format(res_bestFit.loc[c, 0]))
    ax_val.plot(dic_sens[c][name], lw=0, marker='.', fillstyle='none', color=arg['color dual 1'], label='data')
    ax_val.plot(res_sensor1.loc[c, 0] * df_sig1_crop, lw=0.75, color=arg['color Pt'],
                label='α {:.2e}'.format(res_sensor1.loc[c, 0]))
    ax_val.plot(res_sensor2.loc[c, 0] * df_sig2_crop, lw=0.75, color=arg['color Pd'],
                label='α {:.2e}'.format(res_sensor2.loc[c, 0]))
    ax_val.legend(fontsize=9, frameon=True, fancybox=True)

    # plot residuals
    ax_dev.plot(df_sum - dic_sens[c][name], lw=0, marker='.', fillstyle='none', color=arg['color dual 1'])
    ax_dev.axhline(0, color='k', lw=0.5)

    # ------------------------------------------------------------------
    # layout axes
    ax_val.tick_params(axis='both', which='both', direction='out')
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)
    ax_dev.tick_params(axis='both', which='both', direction='out')
    axR1.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT1.tick_params(axis='both', which='both', direction='in', labelsize=0)

    # label axes
    ax_dev.set_xlabel('Wavelength [nm]')
    ax_val.set_ylabel('Norm. intensity [rfu]')
    ax_dev.set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()

    return fig_val


def plot_fitresults_lc(doverlay_crop, result, ar_sig, arg, title=None):
    fig = plt.figure()
    ax = plt.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)
    ax.set_title(title, loc='left')

    ax_dev = plt.subplot2grid((3, 1), (2, 0), colspan=2)
    ax_dev.get_shared_x_axes().join(ax_dev, ax)
    axR_dev = ax_dev.twiny()
    axT_dev = ax_dev.twinx()
    axR_dev.get_shared_x_axes().join(axR_dev, ax_dev)
    axT_dev.get_shared_x_axes().join(axT_dev, ax_dev)

    df_sum1 = result[0] * ar_sig[0][ar_sig[0].columns[0]]
    df_sum2 = result[1] * ar_sig[1][ar_sig[1].columns[0]]
    df_sum_all = df_sum1 + df_sum2

    ax.plot(doverlay_crop, color=arg['color dual 1'], lw=0, marker='o', fillstyle='none')
    ax.plot(doverlay_crop.index, df_sum_all, color='k', ls='--')
    ax.plot(doverlay_crop.index, df_sum1, color=arg['color Pt'], label='a(Pt) = {:.2f}'.format(result[0]))
    ax.plot(doverlay_crop.index, df_sum2, color=arg['color Pd'], label='b(Pd) = {:.2f}'.format(result[1]))
    ax.legend()

    res = doverlay_crop - df_sum_all
    ax_dev.plot(doverlay_crop.index, res, marker='.', color=arg['color dual 1'], lw=0)
    ax_dev.axhline(res.mean(), color='k', lw=0.75)

    ax.tick_params(axis='both', which='both', direction='out')
    axR.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT.tick_params(axis='both', which='both', direction='in', labelsize=0)
    ax_dev.tick_params(axis='both', which='both', direction='out')
    axR_dev.tick_params(axis='both', which='both', direction='in', labelsize=0)
    axT_dev.tick_params(axis='both', which='both', direction='in', labelsize=0)

    ax_dev.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Norm. intensity [a.u.]')
    ax_dev.set_ylabel('Residuals')

    plt.tight_layout()

    return fig


def plot_fitresults_lc_all(data, result, ar_sig, arg, unit, threshold=3.):
    sensID = [c.columns[0].split('-')[0].split(' ')[0] for c in ar_sig]
    nrow = math.ceil(len(data.columns) / 4)
    ncol = 4
    fs = 15
    fig_val, ax_val = plt.subplots(nrows=nrow, ncols=ncol, sharex=True)

    for r in range(nrow):
        for c in range(ncol):
            axR = ax_val[r][c].twinx()
            axT = ax_val[r][c].twiny()
            axR.get_shared_y_axes().join(axR, ax_val[r][c])
            axT.get_shared_x_axes().join(axT, ax_val[r][c])
            ax_val[r][c].tick_params(axis='both', which='both', direction='out')
            axR.tick_params(axis='both', which='both', direction='in', labelcolor='white')
            axT.tick_params(axis='both', which='both', direction='in', labelcolor='white')

    for c in enumerate(data.columns):
        row = int(c[0] / 4)
        col = c[0] % 4

        # data preparation
        df_sum1 = result[c[0]][0] * ar_sig[0][ar_sig[0].columns[0]]
        df_sum2 = result[c[0]][1] * ar_sig[1][ar_sig[1].columns[0]]
        df_sum_all = pd.concat([df_sum1, df_sum2], axis=1, sort=True).sum(axis=1) #df_sum1 + df_sum2
        if max(data[c[1]] - df_sum_all) > threshold:
            print('WARNING! The spectral deviation of the fit result to the measured data for',
                  'concentration point {:.2f}'.format(c[1][0]),
                  'exceed the default threshold level of {:.2f}‰'.format(threshold))
        # plotting
        ax_val[row][col].plot(data[c[1]], color=arg['color dual 1'], lw=0, marker='o', fillstyle='none',
                              label='data {:.2f}'.format(c[1][0]) + unit)
        ax_val[row][col].plot(data[c[1]].index, df_sum_all, color='k', ls='--', label='Fit linear unmixing')
        ax_val[row][col].plot(data[c[1]].index, df_sum1, color=arg['color ' + sensID[0]],
                              label='a({})'.format(sensID[0]) + ' = {:.2f}'.format(result[c[0]][0]))
        ax_val[row][col].plot(data[c[1]].index, df_sum2, color=arg['color ' + sensID[1]],
                              label='b({})'.format(sensID[1]) + ' = {:.2f}'.format(result[c[0]][1]))
        ax_val[row][col].legend(fontsize=fs * 0.6)

    fig_val.text(0.5, 0.02, 'Wavelength [nm]', ha='center', fontsize=fs)
    fig_val.text(0.02, 0.5, 'Rel. intensity', va='center', rotation='vertical', fontsize=fs)

    # fullscreen plot to ameliorite visibility
    plot_backend = matplotlib.get_backend()
    mng = plt.get_current_fig_manager()
    if 'Tk' in plot_backend:
        mng.resize(*mng.window.maxsize())
    elif 'wx' in plot_backend:
        mng.frame.Maximize(True)
    elif 'Qt' in plot_backend:
        mng.window.showMaximized()

    plt.subplots_adjust(left=.07, right=.99, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15)
    plt.show()

    return fig_val


def plotting_fitresultsdual(dic_SVFit, dic_raw, prefactors_std, name_dyes, arg, ratiometric, analyte, unit, simply):
    if len(name_dyes) == 1:
        m = name_dyes[0]
        fig_fit, ax_fit = plt.subplots(figsize=arg['figure size meas'])
        label = []

        axR = ax_fit.twinx()
        axT = ax_fit.twiny()
        axR.get_shared_y_axes().join(axR, ax_fit)
        axT.get_shared_x_axes().join(axT, ax_fit)

        for em, s in enumerate(dic_SVFit['norm best Fit'][m].columns):
            # calibration points and fit
            p1 = ax_fit.plot(dic_raw[m][s].loc[0] / dic_raw[m][s], marker='o', ms=3, fillstyle='none', lw=0,
                             color=arg['color {}'.format(s.split('-')[0])])
            p = ax_fit.plot(1 / dic_SVFit['norm best Fit'][m][s], ls='-.', color=arg['color {}'.format(s.split('-')[0])])

            label.append(s)
            rep = dic_SVFit['Report'][name_dyes[0]][s]
            if simply is True:
                label.append('χ2 = {:.3f} \n f = {:.3f}, $K_{}$$_{}$ = {:.3f}'.format(rep.redchi, rep.best_values['f'],
                                                                                      'S', 'V', rep.best_values['k']))
            else:
                label.append('χ2 = {:.3f} \n f = {:.3f}, $K_{}$$_{}$$_{}$ = {:.3f}, '
                             '$K_{}$$_{}$$_{}$ = {:.3f}'.format(rep.redchi, rep.best_values['f'], 'S', 'V', '1',
                                                           rep.best_values['k'], 'S', 'V', '2',
                                                           rep.best_values['k']*rep.best_values['m']))
            # y-axis limits
            max_ = max(dic_raw[m][s].loc[0] / dic_raw[m][s]) * 1.1
            if max_ < 5.:
                min_ = -0.05
            else:
                min_ = -0.5
            ax_fit.set_ylim(min_, max_)

        # legend
        fig_fit.legend(p, labels=label, ncol=2, loc=0, frameon=True, fancybox=True, borderaxespad=0.2,
                       fontsize=arg['fontsize meas'] * 0.7)

        # add error bars
        for m in name_dyes:
            for em, s in enumerate(dic_SVFit['norm best Fit'][m].columns):
                er = dic_raw[m][s] / dic_raw[m][s].loc[0]
                d = prefactors_std[m][prefactors_std[m].columns[em]]
                yerr = np.abs((d - d.loc[0]) / d.loc[0])
                ax_fit.errorbar(x=er.index, y=1 / er, yerr=yerr, fmt='o', ms=4, mfc='white',
                                color=arg['color {}'.format(s.split('-')[0])])

        # layout axes
        ax_fit.set_title(m, loc='left')
        if ratiometric is True:
            ax_fit.set_ylabel('Integral $R_0/R$', fontsize=arg['fontsize meas'] * 0.9)
        else:
            ax_fit.set_ylabel('Integral $I_0/I$', fontsize=arg['fontsize meas'] * 0.9)

        ax_fit.tick_params(axis='both', which='both', direction='out')
        axR.tick_params(axis='both', which='both', direction='in', labelsize=arg['fontsize meas'] * 0.8,
                        labelcolor='white')
        axT.tick_params(axis='both', which='both', direction='in', labelsize=0)

        if analyte == 'O2':
            par = '$O_2$'
        else:
            print('Define parameter for plotting in plotting_fitresultsdual')
            par = ''
        if unit == '%air':
            unit_ = '(air) [%]'
        elif unit == 'hPa':
            unit_ = '[hPa]'
        else:
            unit_ = unit
        ax_fit.set_xlabel('Concentration ' + par + ' ' + unit_)
    else:
        fig_fit, ax_fit = plt.subplots(nrows=len(name_dyes), sharex=True)
        label = []
        for en, m in enumerate(name_dyes):
            axR = ax_fit[en].twinx()
            axT = ax_fit[en].twiny()
            axR.get_shared_y_axes().join(axR, ax_fit[en])
            axT.get_shared_x_axes().join(axT, ax_fit[en])

            for em, s in enumerate(dic_SVFit['norm best Fit'][m].columns):
                p1 = ax_fit[en].plot(1 / dic_raw[m][s], marker='o', fillstyle='none', lw=0,
                                     color=arg['color {}'.format(s.split('-')[0])])
                p = ax_fit[en].plot(1 / dic_SVFit['norm best Fit'][m][s],
                                    color=arg['color {}'.format(s.split('-')[0])])

                label.append(s)
                rep = dic_SVFit['Report'][m][s]
                if simply is True:
                    label.append('χ2 = {:.3f} \n f = {:.3f}, $K_{}$$_{}$ = {:.3f}'.format(rep.redchi,
                                                                                          rep.best_values['f'], 'S',
                                                                                          'V', rep.best_values['k']))
                else:
                    label.append('χ2 = {:.3f} \n f = {:.3f}, $K_{}$$_{}$$_{}$ = {:.3f}, '
                                 '$K_{}$$_{}$ = {:.3f}'.format(rep.redchi, rep.best_values['f'], 'S', 'V', '1',
                                                               rep.best_values['k'], 'S', 'V', '2',
                                                               rep.best_values['k'] * rep.best_values['m']))

                max_ = max(1 / dic_raw[m][s]) * 1.1
                if max_ < 5:
                    min_ = -0.05
                else:
                    min_ = -0.5
                ax_fit[en].set_ylim(min_, max_)

            # layout axes
            ax_fit[en].set_title(m, loc='left')
            ax_fit[en].set_ylabel('Integral $I_0/I$')
            ax_fit[en].tick_params(axis='both', which='both', direction='out')
            axR.tick_params(axis='both', which='both', direction='in', labelsize=10, labelcolor='white')
            axT.tick_params(axis='both', which='both', direction='in', labelsize=0)

        # legend
        fig_fit.legend([p1, p], labels=label, ncol=2, loc=0, frameon=True, fancybox=True, borderaxespad=0.1)
        ax_fit[-1].set_xlabel('Concentration $O_2$ (air) [%]')

    plt.tight_layout()
    plt.show()

    return fig_fit


def plotting_image_overlay(name_dyes, Xcube, Ycube, cube_plot, dic_calc, frame, dic_calib, max_calib, cutoff,
                           sensortype='single', indicator=None, cmap='inferno'):
    # Plotting the image overlaid with results
    fig_im, ax_im = plt.subplots()
    if indicator == None:
         pass
    else:
         ax_im.set_title(indicator, fontsize=13, x=0.15)

    # cube as background in grey
    ax_im.pcolormesh(Xcube, Ycube, cube_plot, norm=colors.Normalize(vmin=cube_plot.min(), vmax=cube_plot.max()),
                     cmap='gist_yarg')

    # overlaid sensor optodes
    for m in name_dyes:
        if sensortype == 'single':
            if isinstance(max_calib, bool):
                if max_calib is True:
                    vmax = max(dic_calib['data']['pre processed data'].keys()) * (1 + cutoff / 100)
                else:
                    vmax = dic_calc[m].max().max()
            elif isinstance(max_calib, (float, int)):
                vmax = max_calib
            else:
                vmax = 100.
            W, H = np.meshgrid(dic_calc[m].columns, dic_calc[m].index)
            o2_calc = np.array(dic_calc[m])
            data_m = np.ma.masked_where(np.isnan(o2_calc), o2_calc)
            pcm = ax_im.pcolormesh(W, H, data_m, norm=colors.Normalize(vmin=0, vmax=max(vmax)), cmap=cmap)
        elif sensortype == 'multi':
            if isinstance(max_calib, bool):
                if max_calib is True:
                    vmax = max(dic_calib['data']['data'].keys()) * (1 + cutoff / 100)
                else:
                    vmax = dic_calc[m][indicator].max().max()
            elif isinstance(max_calib, (float, int)):
                vmax = max_calib
            else:
                vmax = 100.
            W, H = np.meshgrid(dic_calc[m][indicator].columns, dic_calc[m][indicator].index)
            o2_calc = np.array(dic_calc[m][indicator])
            data_m = np.ma.masked_where(np.isnan(o2_calc), o2_calc)
            pcm = ax_im.pcolormesh(W, H, data_m, norm=colors.Normalize(vmin=0, vmax=vmax), cmap=cmap)
        else:
            raise ValueError('Define sensor type whether it is single or multi analyte sensor')

    # add z-values (analyte concentration) to frame
    W, H = np.meshgrid(frame.columns, frame.index)
    Xflat, Yflat, Zflat = W.flatten(), H.flatten(), frame.values.flatten()

    def fmt(x, y):
        # get closest point with known data
        dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
        idx = np.argmin(dist)
        z = Zflat[idx]
        return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)

    ax_im.format_coord = fmt

    # color bar and inversion of y-axis
    fig_im.colorbar(pcm, ax=ax_im, extend='max')
    # plt.gca().invert_yaxis()

    plt.tight_layout()

    return fig_im


def plotting_image_overlay_joint(meas_dyes, Xcube, Ycube, cube_plot, dic_calc, dic_px90, sensortype='single',
                                 indicator=None, cmap='inferno'):
    # Plotting the image overlaid with results
    fig_im, ax_im = plt.subplots()
    if indicator == None:
         pass
    else:
         ax_im.set_title(indicator, fontsize=13, x=0.15)

    # cube as background in grey
    ax_im.pcolormesh(Xcube, Ycube, cube_plot, norm=colors.Normalize(vmin=cube_plot.min(), vmax=cube_plot.max()),
                     cmap='gist_yarg')

    # overlaid sensor optodes
    for m in meas_dyes:
        if sensortype == 'single':
            o2_calc = np.array(dic_calc[m].T)
        else:
            o2_calc = np.array(dic_calc[m][indicator])
        data_m = np.ma.masked_where(np.isnan(o2_calc), o2_calc)
        W, H = np.meshgrid(dic_px90[m]['px_w'], dic_px90[m]['px_h'])
        pcm = ax_im.pcolormesh(W, H, data_m, norm=colors.Normalize(vmin=0, vmax=100), cmap=cmap)

        Xflat, Yflat, Zflat = W.flatten(), H.flatten(), o2_calc.flatten()

        def fmt(x, y):
            # get closest point with known data
            dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
            idx = np.argmin(dist)
            z = Zflat[idx]
            return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
        ax_im.format_coord = fmt

    # color bar and inversion of y-axis
    fig_im.colorbar(pcm, ax=ax_im, extend='max')
    plt.gca().invert_yaxis()

    plt.tight_layout()

    return fig_im


def plotting_optodes(indicators_calib, meas_dyes, dic_calib, frame, dic_calc, unit, analyte, sensortype='single',
                     indicator=None, cmap='inferno', ncols=2):
    # Plotting only the optode(s)
    c = 0
    nrows = int(len(meas_dyes) % ncols) + ncols
    fig_op, ax_op = plt.subplots(figsize=(7, 7), ncols=ncols, nrows=nrows)

    for en, m in enumerate(meas_dyes):
        if indicators_calib[0] in m:
            en_r = 0
            c += 1
            c = int(c % nrows)
        else:
            en_r = 1
            c += 1
            c = int(c % nrows)

        if sensortype == 'single':
            mean_calc = dic_calc[m].mean().mean()
            std_calc = dic_calc[m].std().std()
            vmax = dic_calib['data']['pre processed data'].keys()
        else:
            mean_calc = dic_calc[m][indicator].mean().mean()
            std_calc = dic_calc[m][indicator].std().std()
            vmax = max(dic_calib['data']['data'].keys())

        if analyte == 'O2':
            para = '$O_2$'
        else:
            print('Define parameter for subtitle')
            para = ''
        ax_op[c][en_r].set_title(m + '\n ' + para + '(' + unit + ') ~ {:.2f} ± {:.2f}%'.format(mean_calc, std_calc),
                                 loc='left')

        W, H = np.meshgrid(dic_calc[m].columns, dic_calc[m].index)
        if sensortype == 'single':
            o2_calc = np.array(dic_calc[m])
        else:
            o2_calc = np.array(dic_calc[m][indicator])
        m = np.ma.masked_where(np.isnan(o2_calc), o2_calc)
        pcm = ax_op[c][en_r].pcolormesh(W, H, m, norm=colors.Normalize(vmin=0, vmax=max(vmax)), cmap=cmap)

        ax_op[c][en_r].invert_yaxis()
        fig_op.colorbar(pcm, ax=ax_op[c][en_r], extend='max')

        W, H = np.meshgrid(frame.columns, frame.index)
        Xflat, Yflat, Zflat = W.flatten(), H.flatten(), frame.values.flatten()

        def fmt(x, y):
            # get closest point with known data
            dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
            idx = np.argmin(dist)
            z = Zflat[idx]
            return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)

        ax_op[c][en_r].format_coord = fmt
    plt.tight_layout()
    plt.show()

    return fig_op


def plotting_dualIndicator(file_meas, meas_dyes, name_single, cube_corr, dic_o2_calc, frame, arg, dic_calib, simply,
                           max_calib, cutoff, unit='%air', analyte='$O_2$', plotting=True, cmap='inferno'):
    # preparation
    img = open_image(file_meas)[:, :, int(np.float(cube_corr['Cube']['cube'].metadata['bands']) - 1)]
    img_rot = corr.rotation_cube(img_cube=img, arg=arg)
    if arg['rotation'] == 0 or arg['rotation'] == 360:
        px_y = np.arange(0, img_rot.shape[0])
        px_x = np.arange(0, img_rot.shape[1])
        cube_plot = img.reshape(img_rot.shape[0], img_rot.shape[1])
        Xcube, Ycube = np.meshgrid(px_x, px_y)
    elif arg['rotation'] == 90:
        px_y = np.arange(0, img_rot.shape[1])
        px_x = np.arange(0, img_rot.shape[0])

        cube_plot = img.reshape(img_rot.shape[1], img_rot.shape[0])
        Ycube, Xcube = np.meshgrid(px_x, px_y)
    elif arg['rotation'] == 180:
        img_ = np.flip(img_rot, 0)
        img = np.flip(img_, 1)
        px_y = np.arange(0, img.shape[0])
        px_x = np.arange(0, img.shape[1])
        cube_plot = img.reshape(img.shape[0], img.shape[1])
        Xcube, Ycube = np.meshgrid(px_x, px_y)
    elif arg['rotation'] == 270:
        img_ = img_rot.swapaxes(0, 1)
        px_y = np.arange(0, img_.shape[1])
        px_x = np.arange(0, img_.shape[0])

        cube_plot = img.reshape(img_.shape[1], img_.shape[0])
        Ycube, Xcube = np.meshgrid(px_x, px_y)
    else:
        Xcube = None
        Ycube = None
        cube_plot = None
    # Plotting the image overlaid with results
    plt.ioff()
    dfig_im = dict()
    for s in name_single:
        fig_im = plotting_image_overlay(name_dyes=meas_dyes, Xcube=Xcube, Ycube=Ycube, cmap=cmap, frame=frame[s],
                                        cube_plot=cube_plot, dic_calc=dic_o2_calc, indicator=s, sensortype='multi',
                                        dic_calib=dic_calib, max_calib=max_calib, cutoff=cutoff)
        dfig_im[s] = fig_im

        if plotting is True:
            plt.show()
        else:
            plt.close(fig_im)

    # ----------------------------------------------------
    # Plotting only the optode(s)
    plt.ioff()
    dic_figures = dict()
    for en, s in enumerate(name_single):
        if len(meas_dyes) == 1:
            fig_op, ax_op = plt.subplots(figsize=(7, 7))
            fig_op.suptitle(s, fontsize=13, x=.5, y=.95)

            ax_op.set_aspect(aspect='auto')
            title = analyte + ' - {:.2f} ± {:.2f} '.format(dic_o2_calc[meas_dyes[0]][s].mean().mean(),
                                                           dic_o2_calc[meas_dyes[0]][s].std().std()) + unit
            ax_op.set_title(title, fontsize=10, x=0.35)

            W, H = np.meshgrid(dic_o2_calc[meas_dyes[0]][s].columns, dic_o2_calc[meas_dyes[0]][s].index)
            o2_calc = np.array(dic_o2_calc[meas_dyes[0]][s])

            if isinstance(max_calib, bool):
                if max_calib is True:
                    vmax = max(dic_calib['data']['data'].keys()) * (1 + cutoff / 100)
                else:
                    vmax = dic_o2_calc[meas_dyes[0]][s].max().max()
            elif isinstance(max_calib, (float, int)):
                vmax = max_calib
            else:
                vmax = 100.
            pcm = ax_op.pcolor(W, H, o2_calc, norm=colors.Normalize(vmin=0, vmax=vmax), cmap=cmap)

            fig_op.colorbar(pcm, ax=ax_op, extend='max')
            ax_op.invert_yaxis()

            W, H = np.meshgrid(frame[s].columns, frame[s].index)
            Xflat, Yflat, Zflat = W.flatten(), H.flatten(), frame[s].values.flatten()

            def fmt(x, y):
                # get closest point with known data
                dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
                idx = np.argmin(dist)
                z = Zflat[idx]
                return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
            ax_op.format_coord = fmt

        elif len(meas_dyes) == 2:
            fig_op, ax_op = plt.subplots(figsize=(7, 7), ncols=2, nrows=math.ceil(len(meas_dyes) / 2))
            fig_op.suptitle(s, fontsize=13, x=.5, y=.95)
            col = 0
            for em, m in enumerate(meas_dyes):
                ax_op[col].set_aspect(aspect='auto')
                title = analyte + ' - {:.2f} ± {:.2f} '.format(dic_o2_calc[m][s].mean().mean(),
                                                               dic_o2_calc[m][s].std().std()) + unit
                ax_op[col].set_title(title, fontsize=10, x=0.35)

                W, H = np.meshgrid(dic_o2_calc[m][s].columns, dic_o2_calc[m][s].index)
                o2_calc = np.array(dic_o2_calc[m][s])

                if isinstance(max_calib, (float, int)):
                    vmax = max_calib
                else:
                    if max_calib is True:
                        vmax = max(dic_calib['data']['data'].keys()) * (1 + cutoff / 100)
                    else:
                        vmax = max(dic_o2_calc[m][s].max().max())
                pcm = ax_op[col].pcolor(W, H, o2_calc, norm=colors.Normalize(vmin=0, vmax=vmax), cmap=cmap)

                fig_op.colorbar(pcm, ax=ax_op[col], extend='max')
                ax_op[col].invert_yaxis()

                W, H = np.meshgrid(frame[s].columns, frame[s].index)
                Xflat, Yflat, Zflat = W.flatten(), H.flatten(), frame[s].values.flatten()

                def fmt(x, y):
                    # get closest point with known data
                    dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
                    idx = np.argmin(dist)
                    z = Zflat[idx]
                    return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)

                ax_op[col].format_coord = fmt
                col += 1

        else:
            fig_op, ax_op = plt.subplots(figsize=(7, 7), ncols=2, nrows=math.ceil(len(meas_dyes) / 2))
            fig_op.suptitle(s, fontsize=13, x=.5, y=.95)

            col = 0
            for em, m in enumerate(meas_dyes):
                row = em % math.ceil(len(meas_dyes) / 2)

                ax_op[row][col].set_aspect(aspect='auto')
                title = analyte + ' - {:.2f} ± {:.2f} '.format(dic_o2_calc[m][s].mean().mean(),
                                                               dic_o2_calc[m][s].std().std()) + unit
                ax_op[row][col].set_title(title, fontsize=10, x=0.35)

                W, H = np.meshgrid(dic_o2_calc[m][s].columns, dic_o2_calc[m][s].index)
                o2_calc = np.array(dic_o2_calc[m][s])

                if isinstance(max_calib, (float, int)):
                    vmax = max_calib
                else:
                    if max_calib is True:
                        vmax = max(dic_calib['data']['data'].keys()) * (1 + cutoff / 100)
                    else:
                        vmax = max(dic_o2_calc[m][s].max().max())
                pcm = ax_op[row][col].pcolor(W, H, o2_calc, norm=colors.Normalize(vmin=0, vmax=vmax), cmap=cmap)

                fig_op.colorbar(pcm, ax=ax_op[row][col], extend='max')
                ax_op[row][col].invert_yaxis()

                W, H = np.meshgrid(frame[s].columns, frame[s].index)
                Xflat, Yflat, Zflat = W.flatten(), H.flatten(), frame[s].values.flatten()
                def fmt(x, y):
                    # get closest point with known data
                    dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
                    idx = np.argmin(dist)
                    z = Zflat[idx]
                    return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
                ax_op[row][col].format_coord = fmt

                if row == math.ceil(len(meas_dyes) / 2) - 1:
                    col += 1

            # if odd number of RoI - remove last subplot in last row
            if (len(meas_dyes) % 2) == 1:
                ax_op[math.ceil(len(meas_dyes) / 2) - 1][1].axis('off')

        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        dic_figures[s] = fig_op
        if plotting is True:
            plt.draw()
        else:
            plt.close(fig_op)
    if plotting is True:
        plt.show()

    return dfig_im, dic_figures


def plot_spectral_deviation(df_ref, data_cube, dye, arg, arg_fit):
    """
    cube_corr requires averaged data
    """
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('Spectral deviation ' + dye, fontsize=10, x=0.365, y=0.96)

    ax = plt.subplot2grid((3, 1), (0, 0), colspan=2, rowspan=2)
    axR = ax.twinx()
    axT = ax.twiny()
    axR.get_shared_y_axes().join(axR, ax)
    axT.get_shared_x_axes().join(axT, ax)

    ax_dev = plt.subplot2grid((3, 1), (2, 0), colspan=2)
    ax_dev.get_shared_x_axes().join(ax_dev, ax)
    axR1 = ax_dev.twiny()
    axT1 = ax_dev.twinx()
    axR1.get_shared_x_axes().join(axR1, ax_dev)
    axT1.get_shared_x_axes().join(axT1, ax_dev)

    # ------------------------------------------------------------------
    # Plotting - emission spectra
    k = 'color ' + dye.split('-')[0]
    if k in arg.keys():
        color = arg[k]
    else:
        color = '#00909e'
    df_val_norm = df_ref[dye] / df_ref[dye].loc[arg_fit['fit range sensor'][0]:].max()
    df_val_norm = df_val_norm.loc[arg_fit['fit range sensor'][0]:]
    ax.plot(df_val_norm, ls='-.', color='grey', label='reference plate-reader')

    data_rel = data_cube.filter(like=dye.split('-')[0]).loc[df_val_norm.index[0]: df_val_norm.index[-1]]
    ax.plot(data_rel / data_rel.max(), color=color, marker='.', fillstyle='none', lw=0, label='HSI fit')

    ax.legend(frameon=True, fancybox=True, fontsize=10)

    # ----------------------------
    # residuals
    ax_dev.axhline(0, color='k', lw=0.75)
    step = int(data_cube.index[1] - data_cube.index[0])
    x = data_rel.filter(like=dye).loc[df_ref.index[0] - step:df_ref.index[-1] + step].index.to_numpy()
    if df_ref.index[0] < x[0]:
        if df_ref.index[-1] > x[-1]:
            xnew = df_ref.loc[x[0]:x[-1]].index.to_numpy()
        else:
            xnew = df_ref.loc[x[0]:].index.to_numpy()
    else:
        if df_ref.index[-1] > x[-1]:
            xnew = df_ref.loc[:x[-1]].index.to_numpy()
        else:
            xnew = df_ref.index.to_numpy()

    y = data_rel.filter(like=dye).loc[df_ref.index[0] - step:df_ref.index[-1] + step][dye].to_numpy()
    finter = interp1d(x, y, kind='cubic')
    data_int = pd.DataFrame(finter(xnew), index=xnew, columns=[dye])
    ax_dev.plot((df_val_norm - data_int[dye]).dropna(), lw=0, marker='.', color='k')

    # ------------------------------------------------------------------
    # layout axes
    ax.tick_params(axis='both', which='both', direction='out')
    axR.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axT.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    ax_dev.tick_params(axis='both', which='both', direction='out')
    axR1.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axT1.tick_params(axis='both', which='both', direction='in', labelcolor='white')

    # label axes
    ax_dev.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Norm. intensity [rfu]')
    ax_dev.set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()


# ==============================================================================================================
def plot_measurement_results_depth_analysis(sens1, sens2, dict_calib, singleID, cutoff, max_calib, cmap='inferno'):
    # sensor 1
    fig_op, ax_op = plt.subplots(figsize=(7, 7))
    fig_op.suptitle(singleID[0], fontsize=12, x=.3, y=0.89)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(1300, 80, 550, 600)  # x, y, dx, dy

    o2_calc1 = np.array(sens1)
    if isinstance(max_calib, bool):
        if max_calib is True:
            vmax = max(dict_calib['calib points']) * (1 + cutoff / 100)
        else:
            vmax = max(sens1.max().max(), sens2.max().max())
    elif isinstance(max_calib, (float, int)):
        vmax = max_calib
    else:
        vmax = 100

    im1 = ax_op.imshow(o2_calc1, interpolation='none', norm=colors.Normalize(vmin=0, vmax=vmax), cmap=cmap)

    # transform px into cm
    no_labels = 7
    ny = o2_calc1.shape[0]
    y_positions = np.linspace(0, ny, num=no_labels) # step between consecutive labels
    p = 0
    ylabel = []
    while p < len(sens1.columns):
        if p < len(sens1.columns):
            ylabel.append(sens1.columns[p])
        p = p + int(len(sens1.columns) / no_labels)
    plt.yticks(y_positions, ylabel)

    no_labels = 5
    nx = o2_calc1.shape[1]
    x_positions = np.linspace(0, nx, num=no_labels) # step between consecutive labels
    p = 0
    xlabel = []
    while p < len(sens1.index):
        if p < len(sens1.index):
            xlabel.append(sens1.index[p])
        p = p + int(len(sens1.index) / no_labels)
    plt.xticks(x_positions, xlabel)
    ax_op.invert_yaxis()

    ax_op.format_coord = Formatter(im1)
    cbar = fig_op.colorbar(im1, ax=ax_op, extend='max')
    cbar.ax.set_ylabel('$O_2$ [hPa]', rotation=0, y=1)

    plt.tight_layout(rect=[0.025, 0.01, 0.95, 0.99], pad=0.15)
    plt.show()

    # ------------------------------------------------------------------------
    # sensor 2
    fig_op2, ax_op2 = plt.subplots(figsize=(7, 7))
    fig_op2.suptitle(singleID[1], fontsize=12, x=.3, y=0.89)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(1300, 480, 550, 600)  # x, y, dx, dy

    o2_calc2 = np.array(sens2)
    im2 = ax_op2.imshow(o2_calc2, interpolation='none', norm=colors.Normalize(vmin=0, vmax=vmax), cmap=cmap)

    # transform px into cm
    no_labels = 7
    ny = o2_calc2.shape[0]
    y_positions = np.linspace(0, ny, num=no_labels) # step between consecutive labels
    p = 0
    ylabel = []
    while p< len(sens2.columns):
        ylabel.append(sens2.columns[p])
        p = p + int(len(sens2.columns) / no_labels)
    plt.yticks(y_positions, ylabel)

    no_labels = 5
    nx = o2_calc2.shape[1]
    x_positions = np.linspace(0, nx, num=no_labels) # step between consecutive labels
    p = 0
    xlabel = []
    while p< len(sens2.index):
        xlabel.append(sens2.index[p])
        p = p + int(len(sens2.index) / no_labels)
    plt.xticks(x_positions, xlabel)
    ax_op2.invert_yaxis()

    ax_op2.format_coord = Formatter(im2)
    cbar = fig_op2.colorbar(im2, ax=ax_op2, extend='max')
    cbar.ax.set_ylabel('$O_2$ [hPa]', rotation=0, y=1)

    # select RoI for depth profile
    toggle_selector.RS = RectangleSelector(ax_op2, onselect, drawtype='line', button=[1, 2], spancoords='pixels',
                                           interactive=True, lineprops=dict(color='white', linestyle='-',
                                                                            linewidth=5, alpha=0.75))
    fig_op2.canvas.mpl_connect('key_press_event', toggle_selector)

    plt.tight_layout(rect=[0.025, 0.01, 0.95, 0.99], pad=0.15) # left, bottom, right, top
    plt.show()

    return toggle_selector, fig_op, fig_op2


def plot_depthProfile(singleID, depth1, depth2, plt_limit=None):
    fig, ax_depth = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(7, 7))
    axR = ax_depth[0].twinx()
    axT = ax_depth[0].twiny()
    axR1 = ax_depth[1].twinx()
    axT1 = ax_depth[1].twiny()
    axR.get_shared_y_axes().join(axR, ax_depth[0])
    axR1.get_shared_y_axes().join(axR1, ax_depth[1])
    axT.get_shared_x_axes().join(axT, ax_depth[0])
    axT1.get_shared_x_axes().join(axT1, ax_depth[1])

    ax_depth[0].set_title(singleID[0], loc='left')
    ax_depth[1].set_title(singleID[1], loc='left')

    ax_depth[0].plot(depth1.index, depth1[0].to_numpy(), color='k', ls='-', lw=1.25)
    ax_depth[1].plot(depth2.index, depth2[0].to_numpy(), color='grey', ls='-', lw=1.25)

    # invert y-axis if the selection was made from the other side
    if depth1[0].to_numpy()[0] > depth1[0].to_numpy()[-1]:
        ax_depth[0].invert_yaxis()
    else:
        pass

    # axis-layout
    if plt_limit is None:
        ax_depth[1].set_xlim(int(depth1.index.min() * 1.1), int(depth1.index.max() * 1.05))
    else:
        ax_depth[1].set_xlim(plt_limit[0], plt_limit[1])
    ax_depth[0].set_ylabel('Depth [cm]')
    ax_depth[1].set_ylabel('Depth [cm]')
    ax_depth[1].set_xlabel('Concentration $O_2$ [hPa]')
    ax_depth[0].tick_params(axis='both', which='both', direction='out')
    ax_depth[1].tick_params(axis='both', which='both', direction='out')
    axR.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axR1.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axT.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    axT1.tick_params(axis='both', which='both', direction='in', labelcolor='white')
    plt.tight_layout()

    return fig, ax_depth