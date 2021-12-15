__author__ = 'Silvia E Zieger'
__project__ = 'sensor response in silico'

"""Copyright 2021. All rights reserved.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable 
for any damages arising from the use of this software.
Permission is granted to anyone to use this software within the scope of evaluating mutli-analyte sensing. No permission
is granted to use the software for commercial applications, and alter it or redistribute it.

This notice may not be removed or altered from any distribution.
"""

import matplotlib
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sp

# --------------------------------------------------------------------------------------------------------------------
# global parameter
n = 1
R = 8.314         # [J/mol*K]
F = 96485         # [C/mol]
s_nernst = 2.303
Tconvert = 273.15

dcolor = dict({'pH': '#1CC49A', 'sig pH': '#4B5258', 'NH3': '#196E94', 'sig NH3': '#314945',
               'NH4': '#DCA744', 'sig NH4': '#89621A', 'TAN': '#A86349'})
ls = dict({'target': '-.', 'simulation': ':'})
fs = 11


# --------------------------------------------------------------------------------------------------------------------
def _gompertz_curve_v1(x, t90, tau, pstart, s_diff, slope='increase'):
    """ Sensor response curve:
    :param x:       np.array; time range
    :param t90:     float; response time of the sensor in s
    :param tau:     float; resolution of the sensor
    :param pstart:  float; starting signal of the response curve
    :param s_diff:  float; difference in signal between end signal and start signal of the response curve
    :param slope:   str; 'increase' or 'decline' describing whether the response increases or decreases
    :return:
    """
    b = np.log(-np.log(tau))
    k = 1 / t90 * (b - np.log(np.log(1 / 0.9)))
    if slope == 'increase':
        y = s_diff * np.exp(-1 * np.exp(b - k*x)) + pstart
    elif slope == 'decline':
        y = pstart - s_diff * np.exp(-1 * np.exp(b - k*x))
    else:
        raise ValueError('Define slope of the response curve either as increase or decline')
    return y


def Nernst_equation(ls_ph, E0, T=25):
    """ convert apparent pH into electrical potential. The logarithmic of the activity equals the pH.
    :param ls_ph:
    :param E0:
    :param T:
    :return:
    """
    faq = s_nernst * R * (T + Tconvert) / (n * F)
    return 1000 * (E0 - faq * ls_ph)


def Nernst_equation_invert(E, E0, T=25):
    return (E0 - E / 1000) * n * F / (2.303 * R * (T + 273.15))


def lin_regression(cnh3_min, cnh3_max, sigNH3_bgd, sigNH3_max, conc_step=0.1):
    cnh3_cali = np.linspace(cnh3_min, cnh3_max, num=int((cnh3_max - cnh3_min) / conc_step + 1))
    ynh3_cali = np.linspace(sigNH3_bgd, sigNH3_max, num=len(cnh3_cali))

    arg = sp.stats.linregress(x=cnh3_cali, y=ynh3_cali)
    return arg


def _calibration_nh3(anh3_min, anh3_max, anh3_step, sigNH3_bgd, sigNH3_max):
    # linear regression for all pH values
    nh3_cali = lin_regression(cnh3_min=anh3_min, cnh3_max=anh3_max, conc_step=anh3_step, sigNH3_bgd=sigNH3_bgd,
                              sigNH3_max=sigNH3_max)
    return nh3_cali


def henderson_nh3(pH, c_nh4, pKa=9.25):
    """ Returns the NH3 concentration depending on the pH and the concentrations of NH4+ """
    return c_nh4 * 10 ** (pH - pKa)


def henderson_nh4(pH, c_nh3, pKa=9.25):
    """ Returns the NH3 concentration depending on the pH and the concentrations of NH4+ """
    return c_nh3 * 10 ** (pKa - pH)


# --------------------------------------------------------------------------------------------------------------------
def _target_fluctuation(ls_conc, tstart, tstop, nP=1):
    """
    Function for periodic block-change of concentration
    :param cnh4_low:  minimal ammonium concentration
    :param cnh4_high: maximal ammonium concentration
    :param tstart:    start time of the period (in s)
    :param tstop:     stop time of the period (in s)
    :param nP:        frequency; number of cycles/period
    :param N:         sample count; a multitude of the time period given
    :param D:         width of pulse; usually half of the period P
    :return:
    """

    N = ((tstop - tstart) / 0.05) + 1
    P = N / nP
    D = P / 2

    # when ls_conc has more than 1 entry -> step function
    if isinstance(ls_conc, tuple):
        ls_sig = list(map(lambda n: list((np.arange(N) % P < D) * (ls_conc[n] - ls_conc[n + 1]) + ls_conc[n + 1]),
                          range(len(ls_conc) - 1)))

        dsig = dict()
        for i in range(len(ls_sig)):
            x = np.linspace(tstop / 2 * i, tstop + tstop / 2 * i, num=int(N))
            dsig[i] = pd.DataFrame(ls_sig[i], columns=['signal'], index=x)

        df_target = pd.concat(dsig, axis=0)
        trange = [i[1] for i in df_target.index]
        df_target.index = trange
    else:
        trange = np.linspace(0, tstop, num=int(N))
        df_target = pd.DataFrame([ls_conc] * len(trange), index=trange, columns=['signal'])

    df_target.index.name = 'time/s'

    return df_target, trange, D


def _target_concentration(ls_ph, ls_cNH3_ppm, ls_cNH4_ppm, t_plateau):
    # always pH
    [target_ph, trange, D] = _target_fluctuation(ls_conc=ls_ph, tstart=0, tstop=t_plateau * 2, nP=1)

    # define analyte - whether NH3 or NH4+
    if np.all([n == None for n in ls_cNH3_ppm]) and not np.all([n == None for n in ls_cNH4_ppm]):
        analyte, ls_nhx = 'NH4', ls_cNH4_ppm
    elif not np.all([n == None for n in ls_cNH3_ppm]) and np.all([n == None for n in ls_cNH4_ppm]):
        analyte, ls_nhx = 'NH3', ls_cNH3_ppm
    else:
        print('ERROR - define a concentration of NH3 / NH4 you want to study')
        print('predefined NH3: 100ppm')
        ls_nhx = 100
        analyte = 'NH3'

    # specify NHx concentrations - consider all options
    ls_nh = tuple()
    for en, n in enumerate(ls_nhx):
        if n != None:
            tn = (n,)
            ls_nh += tn
    if len(ls_nh) == 1:
        ls_nh = ls_nh[0]
    [target_nhx, trange, D] = _target_fluctuation(ls_conc=ls_nh, tstart=0, tstop=trange[-1], nP=1)

    return D, target_ph, target_nhx, analyte


def signal_drift(xtime, df_mV, psensor):
    df_drift = pd.DataFrame(xtime * psensor['drift'] + df_mV['potential mV'].to_numpy())
    df_drift.columns, df_drift.index = ['potential mV'], xtime
    return df_drift


def _sensor_response(df_mV, psensor):
    # what are the specific signal levels
    conc_plateau = list(dict.fromkeys(df_mV['potential mV'].to_numpy()))

    # timing - define sensor time (for simulation of response)
    tplateau = df_mV[df_mV == conc_plateau[0]].dropna().index[-1]
    start, steps = 0, 0.05
    num = int((tplateau - start) / 0.05 + 1)
    sens_time = np.linspace(start, tplateau, num=num)

    # simulate for each signal level a respective sensor response
    c_apparent = psensor['background signal']
    dsig = dict()
    for en, ctarget in enumerate(conc_plateau):
        sdiff = ctarget - c_apparent
        xnew = np.linspace(start + steps * en, tplateau * (en + 1) + steps * en, num=num)

        df_sig = pd.DataFrame(_gompertz_curve_v1(x=sens_time, t90=psensor['t90'], s_diff=sdiff, pstart=c_apparent,
                                                 tau=psensor['resolution'], slope='increase'), index=xnew,
                              columns=['potential mV'])

        start, c_apparent = df_sig['potential mV'].index[-1], df_sig['potential mV'].to_numpy()[-1]
        dsig[en] = df_sig
        df_sensor = pd.concat(dsig)
        df_sensor.index = [i[1] for i in df_sensor.index]

    # include sensor drift if applicable
    df_drift = signal_drift(xtime=df_sensor.index, df_mV=df_sensor, psensor=psensor)

    return df_sensor, df_drift


# --------------------------------------------------------------------------------------------------------------------
def pH_sensor(target_ph, sensor_ph, para_meas):
    # pH sensor - targeted pH fluctuation in mV
    df_sigpH_mV = pd.DataFrame(Nernst_equation(ls_ph=target_ph['signal'].to_numpy(), T=para_meas['temperature'],
                                               E0=sensor_ph['E0']), index=target_ph.index, columns=['potential mV'])

    # -------------------------------------
    # include sensor response
    df_pHrec, df_pHdrift = _sensor_response(df_mV=df_sigpH_mV, psensor=sensor_ph)

    # -------------------------------------
    # re-calculate pH from potential
    df_recalc = pd.DataFrame(Nernst_equation_invert(E=df_pHdrift['potential mV'], E0=sensor_ph['E0'],
                                                    T=para_meas['temperature']))
    df_recalc.columns = ['pH']
    ind_new = [round(i, 2) for i in df_recalc.index]
    df_recalc.index = ind_new

    return df_sigpH_mV, df_pHrec, df_pHdrift, df_recalc


def NHx_sensor(analyte, target_nhx, sensor_nh3):
    # NHx sensor - targeted concentration fluctuation in mV
    para_nh3 = _calibration_nh3(anh3_min=sensor_nh3['NHx calibration'][0], anh3_max=sensor_nh3['NHx calibration'][1],
                                anh3_step=5, sigNH3_bgd=sensor_nh3['signal min'], sigNH3_max=sensor_nh3['signal max'])
    df_sig_mV = pd.DataFrame(target_nhx * para_nh3[0] + para_nh3[1])
    df_sig_mV.columns = ['potential mV']

    # -------------------------------------
    # include sensor response
    df_nhrec, df_nhdrift = _sensor_response(df_mV=df_sig_mV, psensor=sensor_nh3)

    # -------------------------------------
    # re-calculate NHx from potential
    df_recalc = pd.DataFrame((df_nhdrift - para_nh3[1]) / para_nh3[0])
    df_recalc.columns = [analyte]
    ind_new = [round(i, 2) for i in df_recalc.index]
    df_recalc.index = ind_new

    return df_sig_mV, df_nhrec, df_nhdrift, df_recalc


def _other_analyte(analyte, sensor_nh3, df_target, df_calc):
    if analyte == 'NH3':
        # based on target parameter
        c_nhx = henderson_nh4(pKa=sensor_nh3['pKa'], pH=df_target['pH'].to_numpy(), c_nh3=df_target[analyte].to_numpy())
        df_target['NH4'] = pd.DataFrame(c_nhx, index=df_target.index)
        # based on sensor response
        c_nhx = henderson_nh4(pH=df_calc['pH'].to_numpy(), c_nh3=df_calc[analyte].to_numpy(), pKa=sensor_nh3['pKa'])
        df_calc['NH4'] = pd.DataFrame(c_nhx, index=df_calc.index)

    else:
        # based on target parameter
        c_nhx = henderson_nh3(pKa=sensor_nh3['pKa'], pH=df_target['pH'].to_numpy(), c_nh4=df_target[analyte].to_numpy())
        df_target['NH3'] = pd.DataFrame(c_nhx, index=df_target.index)

        # based on sensor response
        c_nhx = henderson_nh3(pH=df_calc['pH'].to_numpy(), c_nh4=df_calc[analyte].to_numpy(), pKa=sensor_nh3['pKa'])
        df_calc['NH3'] = pd.DataFrame(c_nhx, index=df_calc.index)
    return df_target, df_calc


def _tan_calculation(df_target, df_calc):
    # targeted
    df_target['TAN'] = np.sum(df_target.filter(like='NH'), axis=1)
    df_target = df_target.dropna()

    # based on sensor response
    df_calc['TAN'] = np.sum(df_calc.filter(like='NH'), axis=1)
    return df_target, df_calc


# --------------------------------------------------------------------------------------------------------------------
def move_figure(xnew, ynew):
    mngr = plt.get_current_fig_manager()
    geom = mngr.window.geometry()
    x, y, dx, dy = geom.getRect()
    mngr.window.setGeometry(xnew, ynew, dx, dy)


# def plot_sensorresponse(dfconc, sens_time0, tstart_sens, step, D, par_meas, par_sens, arg_fig, plotCheck=True):
#     if 'figsize' in arg_fig.keys():
#         figsize = arg_fig['figsize']
#     else:
#         figsize = (6, 3)
#     fig, ax = plt.subplots(figsize=figsize)
#     if 'title' in arg_fig.keys():
#         fig.canvas.set_window_title(arg_fig['title'])
#     if 'fontsize' in arg_fig.keys():
#         fs = arg_fig['fontsize']
#     else:
#         fs = 10
#     ax.set_xlabel('Time / s', fontsize=fs), ax.set_ylabel('Analyte concentration', fontsize=fs)
#     ax.set_ylim(-0.05, par_sens['max conc']*1.19)
#
#     # ..............................................................
#     # target concentration
#     ax.plot(dfconc, ':k', label='target concentration')
#
#     # time stamps when concentration is changing
#     n, tsample, t = 0, list(), 0
#     while t < par_meas['stop']:
#         t = par_meas['start'] + n * D
#         tsample.append(par_meas['start'] + n * D)
#         n += 1
#
#     # ..............................................................
#     # sensor response
#     # each time I change the concentration, the sensor has to respond to the change
#     # 1) sensor is delayed according to the sampling rate
#     # 2) response does not equal target concentration as sensor has a response time
#     # 3) sensor may have a drift
#     ddata, c_apparent = dict(), dfconc.loc[0, 'signal']
#     for en, t in enumerate(tsample):
#         # closest time step in concentration fluctuation
#         tc = min(dfconc.index, key=lambda x: abs(x - t))
#         if tc != dfconc.index[-1]:
#             # find position in list for previous concentration
#             pos = [en for en in range(len(dfconc.index)) if dfconc.index[en] == tc][0]
#             tc_1, tc1 = dfconc.index[pos - 1], dfconc.index[pos + 1]
#
#             # update actual sensor measurement time +t
#             if en == 0:
#                 sens_time = sens_time0
#             else:
#                 sens_time = np.linspace(sens_time[-1], tc + D, num=int((D-tstart_sens)/step + 1))
#             # sensor signal response
#             df = sensor_response(t=tc1, t_1=tc_1, dfconc=dfconc, par_sens=par_sens, sens_time0=sens_time0,
#                                  c_apparent=c_apparent)
#             df.index = sens_time
#             ddata[tc], c_apparent = df, df['signal'].to_numpy()[-1]
#             ax.plot(sens_time, df, lw=1.25, color='teal', label='sensor response')
#
#             if plotCheck is True:
#                 # check sensor response time
#                 ax.axvline(par_sens['response time'], lw=0.5, color='crimson')
#                 ax.axhline(0.9 * par_sens['max conc'], lw=0.5, color='crimson')
#
#         for t in tsample:
#             # arrows indicating that the sensor starts to react to the surrounding
#             ax.arrow(t, par_sens['max conc']*1.15, 0.0, (par_sens['max conc']-par_sens['max conc']*1.15)/2.5, fc="k",
#                      ec="k", head_width=.8, head_length=0.1)
#     ax.legend(['target concentration', 'sensor response'], loc=0, fontsize=fs * 0.7)
#     plt.tight_layout(), sns.despine(), plt.grid(False)
#
#     # .............................................................................
#     # rearrange data dictionary
#     tstarts = list(ddata.keys())
#     df_data = pd.concat(ddata, axis=0)
#
#     ls_time = list(df_data.loc[tstarts[0]].index)
#     for i in range(len(tstarts) - 1):
#         ls_time.extend(df_data.loc[tstarts[i + 1]].index)
#
#     dfdata = pd.DataFrame(df_data['signal'].to_numpy(), ls_time, columns=['signal'])
#     dfdata.index.name = 'time / s'
#
#     # .............................................................................
#     # sensor drift
#     sig_raw = dfdata
#     sig_drift = pd.DataFrame(sig_raw.index * par_sens['drift'] + sig_raw['signal'].to_numpy())
#     sig_drift.columns = ['signal']
#     sig_drift.index = sig_raw.index
#
#     return fig, sig_raw, sig_drift
#
#
# def plot_2sumresponse(sens_time0, par_target, par_sens1, par_meas, arg_fig, par_sens2=None, plotCheck=False):
#     # individual sensor responses
#     plt.ioff()
#     if isinstance(par_target['conc1'], pd.DataFrame):
#         [fig1, sig_raw1,
#          drift1] = plot_sensorresponse(dfconc=par_target['conc1'], sens_time0=sens_time0, par_meas=par_meas,
#                                        par_sens=par_sens1, step=par_meas['steps'], arg_fig=arg_fig, plotCheck=plotCheck,
#                                        D=par_target['pulse width'], tstart_sens=par_meas['start sensing'])
#     else:
#         fig1, sig_raw1, drift1 = None, None, None
#
#     if isinstance(par_target['conc2'], pd.DataFrame) and par_sens2:
#         [fig2, sig_raw2,
#          drift2] = plot_sensorresponse(dfconc=par_target['conc2'], sens_time0=sens_time0, par_meas=par_meas,
#                                        par_sens=par_sens2, step=par_meas['steps'], plotCheck=plotCheck, arg_fig=arg_fig,
#                                        D=par_target['pulse width'], tstart_sens=par_meas['start sensing'])
#     else:
#         fig2, sig_raw2, drift2 = None, None, None
#     plt.close(fig1), plt.close(fig2)
#
#     # individual analyte over time
#     data = prep_tan_determination(par_sens1=par_sens1, dfdata1=drift1, par_sens2=par_sens2, dfdata2=drift2,
#                                   par_target=par_target, par_meas=par_meas)
#     # target concentration over time
#     data_target = _target_analytes(par_sens1=par_sens1, par_sens2=par_sens2, par_target=par_target, par_meas=par_meas)
#
#     # plotting final simulations
#     figTAN = plot_tanModulation(arg_fig=arg_fig, data=data, data_target=data_target)
#
#     # collect information for output
#     dic_data = dict({'final signal': data, 'raw signal 1': sig_raw1, 'raw signal 2': sig_raw2})
#
#     return figTAN, data_target, dic_data
#
#
# def plot_tanSimulation(df_alpha, phmax=14, pKa=9.25, xnew=50, ynew=60, figsize=(5, 3)):
#     fig, ax = plt.subplots(figsize=figsize)
#     move_figure(xnew=xnew, ynew=ynew)
#     fig.canvas.set_window_title('Ammonia vs. Ammonium')
#     ax.set_xlabel('pH', fontsize=fs), ax.set_ylabel('alpha [%]', fontsize=fs)
#
#     ax.plot(df_alpha['nh4 %'], color=dcolor['NH4'], lw=1., label='NH$_4^+$')
#     ax.plot(df_alpha['nh3 %'], color=dcolor['NH3'], lw=1., label='NH$_3$')
#
#     ax.axvline(pKa, color='k', ls=':', lw=1.)
#     ax.axhline(0.5, color='k', ls=':', lw=1.)
#     ax.legend(loc='upper center', bbox_to_anchor=(1, 0.9), frameon=True, fancybox=True, fontsize=fs * 0.7)
#
#     ax.set_xlim(-0.5, phmax * 1.05)
#     sns.despine(), plt.tight_layout()
#     return fig
#
#
# def plot_tanModulation(arg_fig, data, data_target):
#     figTAN = plt.figure(figsize=arg_fig['figsize'])
#     figTAN.canvas.set_window_title(arg_fig['title'])
#     gs = figTAN.add_gridspec(2,2)
#     ax1 = figTAN.add_subplot(gs[0, 0])
#     ax2 = figTAN.add_subplot(gs[0, 1], sharex=ax1)
#     ax3 = figTAN.add_subplot(gs[1, :], sharex=ax1)
#
#     ax3.set_xlabel('Time / s', fontsize=arg_fig['fontsize']),
#     ax1.set_ylabel('Ind. Analyte concentration / ppm', fontsize=arg_fig['fontsize'])
#     ax3.set_ylabel('sum concentration', fontsize=arg_fig['fontsize'])
#     ax2.set_ylabel('pH')
#
#     # individual sensor signal
#     ax1.plot(data_target.filter(like='nh'), lw=1., ls=':', color='gray')
#     ax1.plot(data.filter(like='nh'), lw=1.)
#     ax2.plot(data_target['pH'], lw=1., ls='--', color='gray')
#     ax2.plot(data.filter(like='pH'), lw=1., color='k')
#
#     # sum parameter compared to target concentration
#     ax3.plot(data['tan'], lw=1., color='#C51D74')
#     ax3.plot(data_target['tan'], lw=1., ls=':', color='k')
#
#     l1 = list(data_target.filter(like='nh').columns)
#     l1.extend(data_target.filter(like='nh').columns)
#     ax1.legend(l1, fontsize=arg_fig['fontsize'] * 0.7, loc='center right')
#
#     ax3.legend(['sum conc.', 'target total conc.'], fontsize=arg_fig['fontsize'] * 0.7, loc='center right')
#     sns.despine(), plt.tight_layout()
#     plt.show()
#     return figTAN
#
#
# def calibration_ph(dfSig_calib, xnew=50, ynew=450, figsize=(5, 3)):
#     fig, ax = plt.subplots(figsize=figsize)
#     move_figure(xnew=xnew, ynew=ynew)
#
#     fig.canvas.set_window_title('Calibration pH sensor (electrochemical)')
#     ax.set_xlabel('pH', fontsize=fs), ax.set_ylabel('Potential / mV', fontsize=fs)
#
#     ax.plot(dfSig_calib, lw=1., color='k')
#     sns.despine(), plt.tight_layout()
#     return fig
#
#
# def calibration_nh3(conc_nh3, para_nh3, xnew=50, ynew=500, figsize=(5, 3)):
#     fig, ax = plt.subplots(figsize=figsize)
#     move_figure(xnew=xnew, ynew=ynew)
#
#     fig.canvas.set_window_title('Calibration NH3 sensor (electrochemical)')
#     ax.set_xlabel('alpha(NH$_3$) / %', fontsize=fs * 0.9), ax.set_ylabel('Sensor signal / mV', fontsize=fs * 0.9)
#
#     ax.plot(conc_nh3, para_nh3[0] * conc_nh3 + para_nh3[1], lw=1., color='k')
#
#     sns.despine(), plt.tight_layout()
#     return fig
#
#
# def plot_tanModel(dfph_re, dfconc_target, df_record, df_tan_target, df_tan, phmax, xnew=690, ynew=350, figsize=(6,4.5)):
#     fig = plt.figure(figsize=figsize)
#     move_figure(xnew=xnew, ynew=ynew)
#     fig.canvas.set_window_title('NH3 / NH4+ simulation')
#
#     gs = GridSpec(nrows=3, ncols=1)
#     ax = fig.add_subplot(gs[:2, 0])
#     ax_ = fig.add_subplot(gs[2, 0], sharex=ax)
#     ax_ph = ax_.twinx()
#     ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
#
#     ax_.set_xlabel('Time / s'), ax_ph.set_ylabel('pH', color='gray')
#     ax_.set_ylabel('TAN / ppm', color=dcolor['TAN'])
#
#     # top plot
#     ax.plot(df_record['nh3 / ppm'], lw=1., color=dcolor['NH3'], label='NH$_3$')
#     ax.plot(df_record['nh4 / ppm'], lw=1., color=dcolor['NH4'], label='NH$_4^+$')
#     ax.legend(frameon=True, fancybox=True, loc=0, fontsize=fs * 0.7)
#     ax.plot(dfconc_target['nh3 / ppm'], lw=1., ls=ls['target'], color='k')
#     ax.plot(dfconc_target['nh4 / ppm'], lw=1., ls=ls['target'], color='gray')
#     ax.set_ylim(-10, dfconc_target['nh3 / ppm'].max()*1.1)
#
#     # bottom plot
#     ax_.plot(df_tan_target, lw=1., ls=ls['target'], color='k')
#     ax_.plot(df_tan, lw=1., color=dcolor['TAN'])
#     ax_ph.plot(dfph_re, lw=1., ls='-.', color='gray')
#     ax_.set_ylim(df_tan['TAN'].min()*0.95, df_tan['TAN'].max()*1.05), ax_ph.set_ylim(-0.5, phmax * 1.05)
#
#     plt.tight_layout()
#     return fig
#
#
# # --------------------------------------------------------------------------------------------------------------------
# def sensor_response(t, t_1, dfconc, par_sens, sens_time0, c_apparent):
#     smax, smin = max(par_sens['min conc'], par_sens['max conc']), min(par_sens['min conc'], par_sens['max conc'])
#     #print(dfconc.loc[t_1, 'signal'], 'vs', smin, dfconc.loc[t, 'signal'], 'vs', smax, c_apparent, 'vs',
#     #      par_sens['max conc'], 'or', par_sens['min conc'])
#     if dfconc.loc[t_1, 'signal'] == smin and dfconc.loc[t, 'signal'] == smin and c_apparent < smax:
#         #print('case1')
#         df = pd.DataFrame(_gompertz_curve_v1(x=sens_time0, t90=par_sens['response time'], pstart=c_apparent,
#                                              tau=1e-5, s_diff=smax - c_apparent, slope='increase'),
#                           index=sens_time0, columns=['signal'])
#     elif dfconc.loc[t_1, 'signal'] == smax and dfconc.loc[t, 'signal'] == smax and c_apparent > smin:
#         #print('case2')
#         df = pd.DataFrame(_gompertz_curve_v1(x=sens_time0, t90=par_sens['response time'], pstart=c_apparent,
#                                              tau=1e-5, s_diff=smax - c_apparent, slope='increase'),
#                           index=sens_time0, columns=['signal'])
#     elif dfconc.loc[t_1, 'signal'] == smin and dfconc.loc[t, 'signal'] == smax:
#         if c_apparent < smax:
#             #print('case3')
#             df = pd.DataFrame(_gompertz_curve_v1(x=sens_time0, t90=par_sens['response time'], pstart=c_apparent, tau=1e-5,
#                                              s_diff=smax - c_apparent, slope='increase'),
#                             index=sens_time0, columns=['signal'])
#         elif c_apparent == smax:
#             df = pd.DataFrame([c_apparent]*len(sens_time0), index=sens_time0, columns=['signal'])
#     elif dfconc.loc[t_1, 'signal'] == smax and dfconc.loc[t, 'signal'] == par_sens['min conc'] and c_apparent < smax:
#         #print('case4')
#         df = pd.DataFrame(_gompertz_curve_v1(x=sens_time0, t90=par_sens['response time'], pstart=c_apparent,
#                                              tau=1e-5, s_diff=c_apparent - smax, slope='decline'),
#                           index=sens_time0, columns=['signal'])
#     elif dfconc.loc[t_1, 'signal'] == smax and dfconc.loc[t, 'signal'] == smin and c_apparent > smin:
#         #print('case5')
#         df = pd.DataFrame(_gompertz_curve_v1(x=sens_time0, t90=par_sens['response time'], pstart=c_apparent,
#                                              tau=1e-5, s_diff=c_apparent-smin, slope='decline'),
#                           index=sens_time0, columns=['signal'])
#     elif dfconc.loc[t_1, 'signal'] == smax and dfconc.loc[t, 'signal'] == smin and c_apparent == smin:
#         #print('case6 - steady state')
#         df = pd.DataFrame([c_apparent]*len(sens_time0), index=sens_time0, columns=['signal'])
#     else:
#         raise ValueError('case undefined at {:.4f}: c(t)={:.4f}, c(t-1)={:.4f}, '
#                          'c_apparent={:.4f}'.format(t, dfconc.loc[t, 'signal'], dfconc.loc[t_1, 'signal'], c_apparent))
#     return df
#
#
# def sensor_response_v1(trange, par_sens, time_sens):
#     pstart, s_diff = par_sens['min conc'], par_sens['max conc'] - par_sens['min conc']
#     dic_response = dict()
#     c, t = 0, 0
#     while t < trange[-1] + 1:
#         df = _gompertz_curve_v1(x=time_sens, t90=par_sens['response time'], tau=1e-6, pstart=pstart, s_diff=s_diff)
#         df_response = pd.DataFrame(df, index=time_sens + c * time_sens[-1], columns=['signal / mV'])
#         dic_response[c] = df_response
#
#         # parameter to connect further
#         pstart = df_response.loc[df_response.index[-1], 'signal / mV']
#         if abs(par_sens['min conc']-pstart) > abs(par_sens['max conc']-pstart):
#             #df_response.loc[df_response.index[-1], 'signal / mV'] == par_sens['max conc']:
#             s_diff = par_sens['min conc'] - par_sens['max conc']
#         else:
#             s_diff = par_sens['max conc'] - par_sens['min conc']
#
#         t = df_response.index[-1] + time_sens[-1]
#         c += 1
#
#     df_response = pd.concat(dic_response)
#     xnew = [i[1] for i in df_response.index]
#     df_response.index = xnew
#
#     return df_response
#
#
# def _pHsensor_response(t_plateau, pH_plateau, dfpH_target, t90_pH, ph_res, sig_bgd, step=0.01):
#     dsig = dict()
#     for n in range(len(pH_plateau)):
#         if n == 0:
#             y_1 = sig_bgd
#         else:
#             print(dsig[n-1])
#             y_1 = dsig[n - 1]['signal / mV'].to_numpy()[-1]
#         sdiff = dfpH_target['target potential / mV'].loc[n * t_plateau] - y_1
#
#         if n == 0:
#             # 1st sensor response - always different
#             t_sens = np.arange(0, t_plateau - 1, step)
#             y0 = _gompertz_curve_v1(x=t_sens, t90=t90_pH, tau=ph_res, pstart=sig_bgd, slope='increase', s_diff=sdiff)
#             dfSig = pd.DataFrame(y0, index=t_sens, columns=['signal / mV'])
#         else:
#             # other sensor responses behave similar
#             t_sens_ = np.arange(0, t_plateau + step, step)
#
#             # different cases - decline or increase
#             if sdiff < 0:
#                 y1 = _gompertz_curve_v1(x=t_sens_, t90=t90_pH, tau=ph_res, pstart=y_1, slope='decline',
#                                         s_diff=np.abs(sdiff))
#             else:
#                 y1 = _gompertz_curve_v1(x=t_sens_, t90=t90_pH, tau=ph_res, pstart=y_1, slope='increase',
#                                         s_diff=np.abs(sdiff))
#             t_sens1 = t_sens_ + n * t_plateau + (n - 1) * step - 1
#             dfSig = pd.DataFrame(y1, index=t_sens1, columns=['signal / mV'])
#         dsig[n] = dfSig
#
#     sens_response = pd.concat(dsig)
#     xnew = sens_response.index.levels[1]
#     sens_response.index = xnew
#     return sens_response
#
#
# def _potential2pH(sens_response, ph_deci, E0, T=25):
#     y = Nernst_equation_invert(E=sens_response['signal / mV'], T=T, E0=E0)
#     dfph_re = pd.DataFrame([round(i, ph_deci) for i in y], index=sens_response.index)
#     dfph_re.columns = ['pH recalc.']
#     return dfph_re
#
#
# def _alpha_vs_time(df_alpha, dfpH_target):
#     # specific target concentration (NH3) over time in percentages
#     c_nh3 = pd.DataFrame(df_alpha['nh3 %'].loc[dfpH_target['pH'].to_numpy()])
#     c_nh4 = pd.DataFrame(df_alpha['nh4 %'].loc[dfpH_target['pH'].to_numpy()])
#     c_nh3.index, c_nh4.index = dfpH_target['pH'].index, dfpH_target['pH'].index
#
#     # DataFrame for NH3/NH4 concentration in percent over time (according to respective pH)
#     dfalpha_target = pd.concat([c_nh3, c_nh4, dfpH_target['pH']], axis=1)
#     return c_nh3, c_nh4, dfalpha_target
#
#
# def conv2potent_nh3(dfalpha_target, c_nh3, c_nh4, dfpH_target, para_nh3):
#     # target concentration over time in ppm
#     dfconc = pd.concat([dfalpha_target['nh3 %'] * c_nh3, dfalpha_target['nh4 %'] * c_nh4,
#                         dfpH_target['pH']], axis=1)
#     dfconc.columns = ['nh3 / ppm', 'nh4 / ppm', 'pH']
#
#     # target potential over time in mV
#     dfpotent = pd.DataFrame(para_nh3[0] * dfconc['nh3 / ppm'] + para_nh3[1])
#     dfpotent.columns, dfpotent.index.name = ['nh3 / mV'], 'time / s'
#     dfpotent['pH'] = dfpH_target['pH']
#
#     return dfconc, dfpotent
#
#
# def _nh3sensor_response(t_plateau, pH_plateau, dfph_re, t90_nh3, nh3_res, dfpot_target, step=.01):
#     dsig_nh3 = dict()
#     for n in range(len(pH_plateau)):
#         if n == 0:
#             y_1 = dfpot_target.loc[0, 'nh3 / mV']
#         else:
#             y_1 = dsig_nh3[n - 1]['signal / mV'].to_numpy()[-1]
#         sdiff = dfpot_target['nh3 / mV'].loc[n * t_plateau] - y_1
#
#         if n == 0:
#             # 1st sensor response - always different
#             t_sens = np.arange(0, t_plateau - 1, step)
#             # apparent pH
#             # ph_i = dfph_re.loc[t_plateau]
#             y1 = _gompertz_curve_v1(x=t_sens, t90=t90_nh3, tau=nh3_res, slope='decline',  s_diff=sdiff,
#                                     pstart=dfpot_target.loc[0, 'nh3 / mV'])
#             dfSig = pd.DataFrame(y1, index=t_sens, columns=['signal / mV'])
#         else:
#             # other sensor responses behave similar
#             t_sens_ = np.arange(0, t_plateau + step, step)
#
#             # different cases - decline or increase
#             if sdiff < 0:
#                 y1 = _gompertz_curve_v1(x=t_sens_, t90=t90_nh3, tau=nh3_res, pstart=y_1, slope='decline',
#                                         s_diff=np.abs(sdiff))
#             else:
#                 y1 = _gompertz_curve_v1(x=t_sens_, t90=t90_nh3, tau=nh3_res, pstart=y_1, slope='increase',
#                                         s_diff=np.abs(sdiff))
#             t_sens1 = t_sens_ + n * t_plateau + (n - 1) * step - 1
#             dfSig = pd.DataFrame(y1, index=t_sens1, columns=['signal / mV'])
#         dsig_nh3[n] = dfSig
#
#     sensNH3_response = pd.concat(dsig_nh3)
#     xnew = sensNH3_response.index.levels[1]
#     sensNH3_response.index = xnew
#     sensNH3_response[dfph_re.columns[0]] = pd.DataFrame(dfph_re, index=xnew)
#     return sensNH3_response
#
#
# def _potent2nh3(sensNH3_response, para_nh3):
#     # apparent pH and respective calibration parameter
#     df = pd.DataFrame((sensNH3_response['signal / mV'] - para_nh3[1]) / para_nh3[0], index=sensNH3_response.index)
#     df.columns = ['nh3 / ppm']
#     return df
#
#
# def _target_analytes(par_sens1, par_sens2, par_target, par_meas):
#     # target analytes
#     dtarget_pH = _assign_target(analyt='pH', par_sens1=par_sens1, par_sens2=par_sens2, par_target=par_target)
#     dtarget_nh4 = _assign_target(analyt='NH4', par_sens1=par_sens1, par_sens2=par_sens2, par_target=par_target)
#     dtarget_nh3 = _assign_target(analyt='NH3', par_sens1=par_sens1, par_sens2=par_sens2, par_target=par_target)
#
#     # recalculate missing analyte
#     if dtarget_nh4 is None:
#         ct_nh4 = henderson_nh4(pKa=par_meas['pKa'], pH=dtarget_pH['signal'].to_numpy(),
#                                c_nh3=dtarget_nh3['signal'].to_numpy())
#         dtarget_nh4 = pd.DataFrame(ct_nh4, columns=['signal'], index=dtarget_nh3.index)
#     elif dtarget_nh3 is None:
#         ct_nh3 = henderson_nh3(pKa=par_meas['pKa'], pH=dtarget_pH['signal'].to_numpy(),
#                                c_nh4=dtarget_nh4['signal'].to_numpy())
#         dtarget_nh3 = pd.DataFrame(ct_nh3, columns=['signal'], index=dtarget_nh4.index)
#
#     data_target = pd.concat([dtarget_pH, dtarget_nh3, dtarget_nh4], axis=1)
#     data_target.columns = ['pH', 'NH3', 'NH4']
#     data_target['TAN'] = np.sum(data_target.filter(like='NH'), axis=1)
#     return data_target
#
#
# def prep_tan_determination(par_sens1, dfdata1, par_sens2, dfdata2, par_target, par_meas):
#     # depending on analytes -> calculate TAN
#     # NH3 and NH4 required
#     [a1_nh3, a1_nh4, a1_pH, data1_nh3, data1_nh4, data1_pH] = assign_analyte(par_sens=par_sens1, dfdata=dfdata1)
#     [a2_nh3, a2_nh4, a2_pH, data2_nh3, data2_nh4, data2_pH] = assign_analyte(par_sens=par_sens2, dfdata=dfdata2)
#     data_nh3, target_nh3 = _assign_data(a1=a1_nh3, a2=a2_nh3, data1=data1_nh3, data2=data2_nh3, dtarget=par_target)
#     data_nh4, target_nh4 = _assign_data(a1=a1_nh4, a2=a2_nh4, data1=data1_nh4, data2=data2_nh4, dtarget=par_target)
#     data_pH, target_pH = _assign_data(a1=a1_pH, a2=a2_pH, data1=data1_pH, data2=data2_pH, dtarget=par_target)
#
#     if data_nh3 is None:
#         c_nh3 = henderson_nh3(pH=data_pH['signal'].to_numpy(), c_nh4=data_nh4['signal'].to_numpy(), pKa=par_meas['pKa'])
#         data_nh3 = pd.DataFrame(c_nh3, columns=['signal'], index=data_nh4.index)
#     elif data_nh4 is None:
#         c_nh4 = henderson_nh4(pH=data_pH['signal'].to_numpy(), c_nh3=data_nh3['signal'].to_numpy(), pKa=par_meas['pKa'])
#         data_nh4 = pd.DataFrame(c_nh4, columns=['signal'], index=data_nh3.index)
#     else:
#         print('something is wrong. At least one analyte is required.')
#
#     # collection of sensor responses
#     data = pd.concat([data_pH, data_nh3, data_nh4], axis=1)
#     data.columns = ['pH', 'NH3', 'NH4']
#     data['TAN'] = np.sum(data.filter(like='NH'), axis=1)
#     return data
#
#
# # --------------------------------------------------------------------------------------------------------------------
# def pH_sensor(sensor_ph, para_meas, df_alpha):
#     # unpacking dictionaries
#     E0, t90_pH, pHres, Tres_ph = sensor_ph['E0'], sensor_ph['t90'], sensor_ph['resolution'], sensor_ph['time steps']
#     pHsens, sig_bgd = sensor_ph['sensitivity'], sensor_ph['background signal']
#     Temp, t_steady, pH_steps = para_meas['Temperature'], para_meas['Plateau time'], para_meas['pH steps']
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # electrochemical pH sensor calibration; source vernier.com/2018/04/06/the-theory-behind-ph-measurements/
#     dfSig_calib = pd.DataFrame(Nernst_equation(ls_ph=np.array(pH_steps), T=Temp, E0=E0), index=pH_steps,
#                                columns=['signal / mV'])
#     dfSig_calib.index.name = 'pH'
#
#     # pH changes to target potential over time
#     # create time range
#     l = [[ph_i] * int(t_steady) for ph_i in pH_steps]
#     ls_pH = l[0]
#     [ls_pH.extend(l_i) for l_i in l[1:]]
#
#     # finalize target pH and concentrations
#     df_ph = pd.DataFrame(ls_pH, columns=['pH'])
#     df_pH = _alpha4ph(df_pH=df_ph, df_alpha=df_alpha)
#
#     dfSig_steps = pd.DataFrame(Nernst_equation(ls_ph=np.array(df_pH['pH'].to_numpy()), T=Temp, E0=E0),
#                                index=df_pH['pH'].index, columns=['target potential / mV'])
#     dfpH_target = pd.concat([df_pH['pH'], dfSig_steps], axis=1)
#
#     # include sensor response
#     sens_response = _pHsensor_response(t_plateau=t_steady, pH_plateau=pH_steps, ph_res=pHres, dfpH_target=dfpH_target,
#                                        t90_pH=t90_pH, step=Tres_ph, sig_bgd=sig_bgd)
#
#     # recalculation of pH values
#     dfph_re = _potential2pH(sens_response=sens_response, ph_deci=pHsens, E0=E0, T=Temp)
#
#     return dfpH_target, dfSig_calib, dfph_re
#
#
# def NH3_sensor(df_alpha, dfph_re, dfpH_target, sensor_nh3, para_meas):
#     # unpacking dictionaries
#     t90_nh3, nh3res, Tsteps_nh3 = sensor_nh3['response time'], sensor_nh3['resolution'], sensor_nh3['time steps']
#     Temp, t_steady, pH_steps = para_meas['Temperature'], para_meas['Plateau time'], para_meas['pH steps']
#
#     pHrange, ph_deci, pKa = sensor_nh3['pH range'], sensor_nh3['sensitivity'], sensor_nh3['pKa']
#     nh3_range, signal_min, signal_max = sensor_nh3['nh3 range'], sensor_nh3['signal min'], sensor_nh3['signal max']
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # linear regression for all pH values
#     para_nh3 = _calibration_nh3(anh3_min=nh3_range[0], anh3_max=nh3_range[1], anh3_step=nh3_range[2],
#                                 sigNH3_bgd=signal_min, sigNH3_max=signal_max)
#
#     # target NH3 signal for NH3 and NH4+ according to the measured pH - specific target concentration over time in %
#     nh3_target, nh4_target, dfalpha_target = _alpha_vs_time(df_alpha=df_alpha, dfpH_target=dfpH_target)
#
#
#     if 'target concentration' in para_meas:
#         cnh3_target = para_meas['target concentration']
#     else:
#         # assume a constant ammonia concentration over time
#         cnh3_target = [para_meas['GGW concentration']]*len(dfalpha_target.index)
#
#     [dfconc_target,
#      dfpot_target] = conv2potent_nh3(dfalpha_target=dfalpha_target, c_nh3=cnh3_target, c_nh4=cnh3_target,
#                                      dfpH_target=dfpH_target, para_nh3=para_nh3)
#
#     # include sensor response - double checked. NH3 starting with ideal situation
#     sensNH3_resp = _nh3sensor_response(t_plateau=t_steady, pH_plateau=pH_steps, step=Tsteps_nh3, dfph_re=dfph_re,
#                                        t90_nh3=t90_nh3, nh3_res=nh3res, dfpot_target=dfpot_target)
#
#     # recalculation of NH3 concentrations while considering the pH(recalc.)
#     dfnh3_calc = _potent2nh3(sensNH3_response=sensNH3_resp, para_nh3=para_nh3)
#
#     # get NH4 concentration via Henderson-Hasselbalch
#     dfnh4_calc = pd.DataFrame(henderson_nh4(pKa=pKa, pH=dfph_re['pH recalc.'].to_numpy(),
#                                             c_nh3=dfnh3_calc['nh3 / ppm'].to_numpy()), columns=['nh4 / ppm'],
#                               index=dfnh3_calc.index)
#
#     # include sampling rate for actually reading out sensor data
#     x_smg = np.arange(dfnh3_calc.index[0], dfnh3_calc.index[-1], para_meas['sampling rate'])
#     x_smg = [i.round(2) for i in x_smg]
#
#     df_record = pd.concat([dfnh3_calc, dfnh4_calc], axis=1)
#     tnew = [round(i, 2) for i in df_record.index]
#     df_record.index = tnew
#     df_record = df_record.loc[x_smg]
#
#     # Calculation total ammonia nitrogen: TAN = NH3 + NH4+
#     df_tan_target = pd.DataFrame(dfconc_target[['nh3 / ppm', 'nh4 / ppm']].sum(axis=1), columns=['TAN'])
#     df_tan = pd.DataFrame(df_record.dropna().sum(axis=1), columns=['TAN'])
#     return df_tan_target, dfconc_target, para_nh3, df_record, df_tan
#
#
# def tan_simulation(sensor_ph, para_meas, sensor_nh3, plot='result'):
#     # unpacking dictionaries
#     pHrange, ph_deci, pKa = sensor_nh3['pH range'], sensor_nh3['sensitivity'], sensor_nh3['pKa']
#     nh3_range, signal_min, signal_max = sensor_nh3['nh3 range'], sensor_nh3['signal min'], sensor_nh3['signal max']
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # TAN system - pH curve of NH3 and NH4 in percentage
#     df_alpha = _tan_simulation(c_nh4=para_meas['GGW concentration'], phmin=pHrange[0], phmax=pHrange[1],
#                                step_ph=pHrange[2], ph_deci=ph_deci, pKa=pKa)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # pH sensor modelation
#     dfpH_target, dfSig_calib, dfph_re = pH_sensor(sensor_ph=sensor_ph, para_meas=para_meas, df_alpha=df_alpha)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # NH3 / NH4+ sensor modulation
#     [df_tan_target, dfconc_target, para_nh3,
#      df_record, df_tan] = NH3_sensor(df_alpha=df_alpha, dfph_re=dfph_re, dfpH_target=dfpH_target, sensor_nh3=sensor_nh3,
#                                      para_meas=para_meas)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # plotting part
#     # Turn interactive plotting off
#     plt.ioff()
#
#     # system simulation
#     fig = plot_tanSimulation(df_alpha=df_alpha, phmax=14, pKa=9.25)
#
#     # single sensor calibration
#     fig1 = calibration_ph(dfSig_calib=dfSig_calib)
#     conc_nh3 = np.arange(nh3_range[0], nh3_range[1], step=nh3_range[2])
#     fig2 = calibration_nh3(conc_nh3, para_nh3, xnew=50, ynew=500, figsize=(5, 3))
#
#     # final model
#     fig3 = plot_tanModel(dfph_re=dfph_re, dfconc_target=dfconc_target, df_record=df_record, df_tan_target=df_tan_target,
#                          df_tan=df_tan, phmax=sensor_nh3['pH range'][1])
#
#     # depending on user input close certain figure plots
#     if plot == 'calib':
#         plt.close(fig3)
#     elif plot == 'results':
#         plt.close(fig), plt.close(fig1), plt.close(fig2)
#     elif plot == 'all':
#         pass
#     # Display all "open" (non-closed) figures
#     plt.show()
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # collect for result output
#     dic_target = dict({'TAN': df_tan_target, 'NH3 simulation': df_alpha, 'target conc nh3': dfconc_target})
#     dic_sens_calib = dict({'pH': dfSig_calib, 'NH3': para_nh3})
#     dic_sens_record = dict({'tan': df_tan, 'NH3': df_record, 'pH': dfph_re})
#     dic_figures = dict({'tan simulation': fig, 'pH calib': fig1, 'NH3 calib': fig2, 'model': fig3})
#
#     return dic_target, dic_sens_calib, dic_sens_record, dic_figures, df_tan
#
#
# def save_report(para_meas, sensor_ph, sensor_nh3, dsens_record, dtarget):
#     df_p = pd.DataFrame(np.zeros(shape=(len(para_meas.values()), 2)))
#     df_p[0] = list(para_meas.keys())
#     df_p[1] = para_meas.values()
#     df_p.loc[-1, :] = ['parameter', 'values']
#     df_p = df_p.sort_index()
#     df_p.columns = ['parameter', 'values']
#     df_p.index = ['general'] * len(df_p.index)
#
#     df_ph = pd.DataFrame(np.zeros(shape=(len(sensor_ph.values()), 2)))
#     df_ph[0] = list(sensor_ph.keys())
#     df_ph[1] = sensor_ph.values()
#     df_ph.columns = ['parameter', 'values']
#     df_ph.index = ['ph'] * len(df_ph.index)
#
#     df_nh3 = pd.DataFrame(np.zeros(shape=(len(sensor_nh3.values()), 2)))
#     df_nh3[0] = list(sensor_nh3.keys())
#     df_nh3[1] = sensor_nh3.values()
#     df_nh3.columns = ['parameter', 'values']
#     df_nh3.index = ['nh3'] * len(df_nh3.index)
#
#     df_para = pd.concat([df_p, df_ph, df_nh3])
#
#     # ..................................................................
#     # results
#     dd = pd.concat([dsens_record['NH3'], dsens_record['tan']], axis=1).T.sort_index().T
#     df_res_ = pd.concat([dd, dsens_record['pH']])
#     df_res_.columns = ['TAN_record', 'nh3_record / ppm', 'nh4_record / ppm', 'pH_record']
#
#     df_ = pd.concat([dtarget['target conc nh3'], dtarget['TAN']], axis=1).T.sort_index().T
#     df_.columns = ['TAN_target', 'nh3_target / ppm', 'nh4_target / ppm', 'pH_target']
#     df_res = pd.concat([df_, df_res_]).sort_index().T.sort_index().T
#     xnew = [int(i) for i in df_res.index]
#     df_res.index = xnew
#     df_res = df_res.groupby(df_res.index).mean()
#     header_res = pd.DataFrame(df_res.columns, columns=['Time / s'], index=df_res.columns).T
#
#     df_out = pd.concat([header_res, df_res])
#     df_para.columns = [0, 1]
#     df_out.columns = np.arange(0, len(df_out.columns))
#
#     output = pd.concat([df_para, df_out])
#
#     return output
#
#
# def load_data(file):
#     file_ = open(file, 'r')
#     count = 0
#     ls_lines = list()
#     while True:
#         count += 1
#         line = file_.readline()
#         # if line is empty end of file is reached
#         if not line:
#             break
#         ls_lines.append(line.strip().split('\t'))
#     file_.close()
#
#     # ............................................................
#     ls_general = list()
#     for l in ls_lines:
#         if l[0] == 'general':
#             ls_general.append(l[1:])
#     df_general = pd.DataFrame(ls_general).T.set_index(0).T.set_index('parameter')
#
#     ls_ph = list()
#     for l in ls_lines:
#         if l[0] == 'ph':
#             ls_ph.append(l[1:])
#     df_ph = pd.DataFrame(ls_ph, columns=['parameter', 'values'])
#     df_ph = df_ph.set_index('parameter')
#
#     ls_nh3 = list()
#     for l in ls_lines:
#         if l[0] == 'nh3':
#             ls_nh3.append(l[1:])
#     df_nh3 = pd.DataFrame(ls_nh3, columns=['parameter', 'values'])
#     df_nh3 = df_nh3.set_index('parameter')
#
#     return df_general, df_ph, df_nh3