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
import numpy as np
import pandas as pd

import curve_fitting_Cube_v1_4 as cC
import correction_hyperCamera_v1_4 as corr
import layout_plotting_v1_3 as plot
import calibration_Cube_v1_5 as calib


# ====================================================================================================================
# plotting oxygen concentration
def regression_line(drawLine):
    # slope k = dy/dx
    dx = -1*(drawLine['start_px'][0] - drawLine['end_px'][0])
    dy = -1*(drawLine['start_px'][1] - drawLine['end_px'][1])

    # abszisse t = y-kx
    if np.abs(dx) >= 2:
        k = dy / dx
        t = drawLine['start_px'][1] - k * drawLine['start_px'][0]
    else:
        k = None
        t = np.linspace(start=drawLine['end_px'][1], stop=drawLine['start_px'][1], endpoint=True,
                        num=int(np.abs(drawLine['start_px'][1] - drawLine['end_px'][1]) + 1))
    return k, t


def convert_px_cm(cm, analyte1, analyte2):
    # cm is a tuple with (px, cm) of a known distance
    # convert px to mm for dataframe
    # px -> cm conversion
    d_px = float(cm.split(',')[1]) / int(cm.split(',')[0])  # px/cm

    # re-arrange matrix px for more comfortable processing
    px_index = np.linspace(0, stop=analyte1.shape[0] - 1, num=analyte1.shape[0], dtype=int)
    px_columns = np.linspace(0, stop=analyte1.shape[1] - 1, num=analyte1.shape[1], dtype=int)

    analyte1_px = analyte1.copy()
    analyte1_px.index = px_index
    analyte1_px.columns = px_columns
    analyte2_px = analyte2.copy()
    analyte2_px.index = px_index
    analyte2_px.columns = px_columns

    # apply px-conversion to measurement results
    sens1 = analyte1_px.copy()
    sens1.columns = [round(f, 3) for f in analyte1_px.columns * d_px]
    sens1.index = [round(f, 3) for f in analyte1_px.index * d_px]

    sens2 = analyte2_px.copy()
    sens2.columns = [round(f, 3) for f in analyte2_px.columns * d_px]
    sens2.index = [round(f, 3) for f in analyte2_px.index * d_px]

    return d_px, sens1, sens2, analyte1_px, analyte2_px


def substitute_cm2px(analyte1, analyte2):
    xnew = reversed(np.linspace(0, stop=analyte1.shape[0], num=analyte1.shape[0], dtype=int))
    ynew = reversed(np.linspace(0, stop=analyte1.shape[1], num=analyte1.shape[1], dtype=int))
    sens1 = analyte1.copy()
    sens1.index = xnew
    sens1.columns = ynew

    xnew = reversed(np.linspace(0, stop=analyte2.shape[0], num=analyte2.shape[0], dtype=int))
    ynew = reversed(np.linspace(0, stop=analyte2.shape[1], num=analyte2.shape[1], dtype=int))
    sens2 = analyte2.copy()
    sens2.index = xnew
    sens2.columns = ynew

    return sens1, sens2


def drawnLine_extract_points(toggle_selector):
    drawLine = dict()
    drawLine['end_px'] = (int(round(toggle_selector.RS.geometry[0][0])), int(round(toggle_selector.RS.geometry[1][0])))
    drawLine['start_px'] = (int(round(toggle_selector.RS.geometry[0][1])), int(round(toggle_selector.RS.geometry[1][1])))

    return drawLine


# def values_alongLine(d_px, analyte1, analyte2=None, toggle_selector=None, drawLine=None, lw_profile=10.):
#     # extract line from plot and determine slope and abscissa
#     if drawLine is None:
#         drawLine = drawnLine_extract_points(toggle_selector)
#     slope, y_absz = regression_line(drawLine)
#     print(drawLine, 'slope', slope, 'abscissa', y_absz)
#
#     # check whether a line for profiling is selected
#     if drawLine is None:
#         print('No profile selected. Please, provide a line profile in the subplot!')
#
#     # select which case
#     if slope is None:
#         print('vertical')
#         i1 = analyte1.loc[y_absz, (drawLine['start_px'][0] -
#                                    lw_profile / 2):(drawLine['start_px'][0] + lw_profile / 2)]
#         if analyte2:
#             i2 = analyte2.loc[y_absz, (drawLine['start_px'][0] -
#                                        lw_profile / 2):(drawLine['start_px'][0] + lw_profile / 2)]
#
#         # extracted data with cm unit
#         i1_av = i1.mean(axis=1)
#         df1 = pd.DataFrame(i1_av.index * d_px, index=i1_av.to_numpy())
#         if analyte2:
#             i2_av = i2.mean(axis=1)
#             df2 = pd.DataFrame(i2_av.index * d_px, index=i2_av.to_numpy())
#     elif isinstance(slope, (int, float)):
#         if slope > 70:
#             print('almost vertical')
#             ydata = np.linspace(start=drawLine['start_px'][1], stop=drawLine['end_px'][1],
#                                 num=np.abs(drawLine['end_px'][1] - drawLine['start_px'][1]) + 1)
#
#             i1 = analyte1.loc[ydata, (drawLine['start_px'][0] -
#                                       lw_profile / 2):(drawLine['start_px'][0] + lw_profile / 2)]
#             if analyte2:
#                 i2 = analyte2.loc[ydata, (drawLine['start_px'][0] -
#                                           lw_profile / 2):(drawLine['start_px'][0] + lw_profile / 2)]
#
#             # extracted data with cm unit
#             i1_av = i1.mean(axis=1)
#             df1 = pd.DataFrame(i1_av.index * d_px, index=i1_av.to_numpy())
#             if analyte2:
#                 i2_av = i2.mean(axis=1)
#                 df2 = pd.DataFrame(i2_av.index * d_px, index=i2_av.to_numpy())
#
#         elif round(slope) == 0:
#             print('horizontal')
#             i1 = analyte1.loc[int(y_absz) - lw_profile / 2:int(y_absz) + lw_profile / 2,
#                  drawLine['end_px'][0]:drawLine['start_px'][0]]
#             if analyte2:
#                 i2 = analyte2.loc[int(y_absz) - lw_profile / 2:int(y_absz) + lw_profile / 2,
#                      drawLine['end_px'][0]:drawLine['start_px'][0]]
#
#             # extracted data with cm unit
#             i1_av = i1.mean()
#             df1 = pd.DataFrame(i1_av.index * d_px, index=i1_av.to_numpy())
#             if analyte2:
#                 i2_av = i2.mean()
#                 df2 = pd.DataFrame(i2_av.index * d_px, index=i2_av.to_numpy())
#         else:
#             print('random line')
#             xdata = np.linspace(start=drawLine['end_px'][0], stop=drawLine['start_px'][0],
#                                 num=int(np.abs(drawLine['end_px'][0] - drawLine['start_px'][0]) + 1),
#                                 dtype=int)  # columns
#             ydata = [int(i) for i in xdata * slope + y_absz]  # index
#
#             i1 = list()
#             i2 = list()
#             for en, x in enumerate(xdata):
#                 i1.append(analyte1.loc[ydata[en], xdata[en]])
#                 if analyte2:
#                     i2.append(analyte2.loc[ydata[en], xdata[en]])
#
#             #  extracted data with cm unit
#             df1 = pd.DataFrame([t * d_px for t in xdata * slope + y_absz], index=i1)
#             if analyte2:
#                 df2 = pd.DataFrame([t * d_px for t in xdata * slope + y_absz], index=i2)
#             else:
#                 i2 = None
#                 df2 = None
#
#     return i1, i2, df1, df2, drawLine, slope, y_absz


def valuesFromLine(d_px, analyte1, analyte2=None, drawLine=None, toggle_selector=None, lw_profile=10.):
    # extract line from plot and determine slope and abscissa
    if drawLine is None:
        drawLine = drawnLine_extract_points(toggle_selector)
    slope, y_absz = regression_line(drawLine)
    print(drawLine, 'slope', slope, 'abscissa', y_absz)

    # check whether a line for profiling is selected
    if drawLine is None:
        print('No profile selected. Please, provide a line profile in the subplot!')

    # --------------------------------------------------------------------------------------------------------------
    # extract values from dataframe -> along line
    i1, df1 = values_anlongLine(d_px=d_px, analyte=analyte1, slope=slope, y_absz=y_absz, drawLine=drawLine,
                                lw_profile=lw_profile)
    if analyte2 is None:
        i2 = None
        df2 = None
    else:
        i2, df2 = values_anlongLine(d_px=d_px, analyte=analyte2, slope=slope, y_absz=y_absz, drawLine=drawLine,
                                    lw_profile=lw_profile)

    return i1, i2, df1, df2, drawLine, slope, y_absz


def values_anlongLine(d_px, analyte, slope, y_absz, drawLine, lw_profile=10.):
    # select which case
    if slope is None:
        print('vertical')
        i1 = analyte.loc[y_absz, (drawLine['start_px'][0] -
                                  lw_profile / 2):(drawLine['start_px'][0] + lw_profile / 2)]

        # extracted data with cm unit
        i1_av = i1.mean(axis=1)
        df1 = pd.DataFrame(i1_av.index * d_px, index=i1_av.to_numpy())
    elif isinstance(slope, (int, float)):
        if slope > 70:
            print('almost vertical')
            ydata = np.linspace(start=drawLine['start_px'][1], stop=drawLine['end_px'][1],
                                num=np.abs(drawLine['end_px'][1] - drawLine['start_px'][1]) + 1)

            i1 = analyte.loc[ydata, (drawLine['start_px'][0] -
                                     lw_profile / 2):(drawLine['start_px'][0] + lw_profile / 2)]

            # extracted data with cm unit
            i1_av = i1.mean(axis=1)
            df1 = pd.DataFrame(i1_av.index * d_px, index=i1_av.to_numpy())

        elif round(slope) == 0:
            print('horizontal')
            i1 = analyte.loc[int(y_absz) - lw_profile / 2:int(y_absz) + lw_profile / 2,
                 drawLine['end_px'][0]:drawLine['start_px'][0]]

            # extracted data with cm unit
            i1_av = i1.mean()
            df1 = pd.DataFrame(i1_av.index * d_px, index=i1_av.to_numpy())
        else:
            print('random line')
            xdata = np.linspace(start=drawLine['end_px'][0], stop=drawLine['start_px'][0],
                                num=int(np.abs(drawLine['end_px'][0] - drawLine['start_px'][0]) + 1),
                                dtype=int)  # columns
            ydata = [int(i) for i in xdata * slope + y_absz]  # index

            i1 = list()
            i2 = list()
            for en, x in enumerate(xdata):
                i1.append(analyte.loc[ydata[en], xdata[en]])

            #  extracted data with cm unit
            df1 = pd.DataFrame([t * d_px for t in xdata * slope + y_absz], index=i1)

    return i1, df1

