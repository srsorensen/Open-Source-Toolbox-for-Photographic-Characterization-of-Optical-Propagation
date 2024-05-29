# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:39:55 2023

@author: frede
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from itertools import zip_longest
import pandas as pd
import klepto

def get_polarization_name(filename):
    filename = filename.split(".")[0]
    polarization = " ".join(filename.split("_")[-2:])
    return polarization


def mean_sweep_data(sweep_data):
    sweep_data = [data[np.isfinite(data)] for data in sweep_data]
    columns = zip_longest(*sweep_data, fillvalue=0)
    return [sum(col) / len(sweep_data) for col in columns]


def weighted_mean(values, weights):
    if len(values) == 1:
        return values[0], weights[0]
    if type(values) == list:
        values = np.array(values)
        weights = np.array(weights)
    weights = 1 / weights
    weighted_mean = np.sum(values * weights) / np.sum(weights)
    weighted_std = np.sqrt(np.sum(weights * (values - weighted_mean) ** 2) / np.sum(weights))
    return weighted_mean, weighted_std


def plot_polarization_data(wavelengths, data, data_wav, polarization_label, color, weights):

    if len(data) > 0:
        dataframe = pd.DataFrame(columns=wavelengths)
        dataframe_weights = pd.DataFrame(columns=wavelengths)
        for i in range(len(data)):
            series = pd.Series(data[i], index=data_wav[i])
            dataframe = pd.concat([dataframe, series.to_frame().T], ignore_index=True)

        for i in range(len(weights)):
            series = pd.Series(weights[i], index=data_wav[i])
            dataframe_weights = pd.concat([dataframe_weights, series.to_frame().T], ignore_index=True)
        #print(dataframe_weights)
        # dataframe.loc[:,cutlist[0]:cutlist[1]] = dataframe.loc[:,cutlist[0]:cutlist[1]].dropna(axis=1)
        dataframe = dataframe.dropna(axis=1)
        dataframe_weights = dataframe_weights.dropna(axis=1)
        for i in range(len(dataframe)):
            alpha_data = dataframe.iloc[i, :]
            wav_data = list(dataframe.iloc[i, :].index)
            plt.plot(wav_data, alpha_data, f"{color}-", label=f"{polarization_label} measurements")

        weighted_average_sweep = []
        weighted_std_sweep = []
        for i in range((dataframe.shape[1])):
            column = dataframe.iloc[:, i]
            col_weights = dataframe_weights.iloc[:, i]
            mean, std = weighted_mean(column, col_weights)
            weighted_average_sweep.append(mean)
            weighted_std_sweep.append(std)

        weighted_average_sweep = pd.Series(weighted_average_sweep, index=dataframe_weights.columns)
        weighted_std_sweep = pd.Series(weighted_std_sweep, index=dataframe_weights.columns)

        return dataframe, weighted_average_sweep, weighted_std_sweep
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def load_data(directory,contains,f_ending,threshold,alpha_threshold,r_squared_threshold):

    TE_data = []
    TM_data = []

    TE_weights = []
    TM_weights = []

    TE_wav = []
    TM_wav = []

    TE_r_squared = []
    TM_r_squared = []

    TE_left_indent = []
    TM_left_indent = []

    TE_sum_width = []
    TM_sum_width = []

    for file in os.listdir(directory):
        if contains.lower() in file.lower() and f_ending.lower() in file.lower():

            hf = h5py.File(directory + file, 'r')
            wav_nm = np.array(hf.get('wavelength'))
            weights = np.array(hf.get('alpha_variance'))
            left_indent = np.array(hf.get('left_indent'))
            sum_width = np.array(hf.get('sum_width'))
            r_squared = np.array(hf.get('r_squared'))
            alphas = np.array(hf.get('alpha'))

            if threshold:
                indices_above_threshold = np.where(alphas > alpha_threshold)[0]
                alphas = alphas[alphas <= alpha_threshold]
                wav_nm = np.delete(wav_nm, indices_above_threshold)
                weights = np.delete(weights, indices_above_threshold)
                left_indent = np.delete(left_indent, indices_above_threshold)
                sum_width = np.delete(sum_width, indices_above_threshold)
                r_squared = np.delete(r_squared, indices_above_threshold)

                indices_above_threshold = np.where(alphas > r_squared_threshold)[0]
                alphas = alphas[alphas <= r_squared_threshold]
                wav_nm = np.delete(wav_nm, indices_above_threshold)
                weights = np.delete(weights, indices_above_threshold)
                left_indent = np.delete(left_indent, indices_above_threshold)
                sum_width = np.delete(sum_width, indices_above_threshold)
                r_squared = np.delete(r_squared, indices_above_threshold)

            y_savgol = savgol_filter(alphas.tolist(), 501, 1, mode="nearest")

            if "TE" in file:
                TE_data.append(list(y_savgol))
                TE_weights.append(weights)
                TE_wav.append(([float(x.decode()) for x in wav_nm]))
                TE_r_squared.append(r_squared)
                TE_left_indent.append(left_indent)
                TE_sum_width.append(sum_width)

            else:
                TM_data.append(list(y_savgol))
                TM_weights.append(weights)
                TM_wav.append(([float(x.decode()) for x in wav_nm]))
                TM_r_squared.append(r_squared)
                TM_left_indent.append(left_indent)
                TM_sum_width.append(sum_width)

            hf.close()

    return TE_data, TE_wav, TE_weights, TM_data, TM_wav, TM_weights

directory = 'D:/Top_Down_Method/Article_Data/Processed data/AlGaAs/Correct_Optimized/'
contains = 'ST3'
f_ending = '.h5'
threshold = False
alpha_threshold = 1000
r_squared_threshold = 0.0

TE_data, TE_wav, TE_weights, TM_data, TM_wav, TM_weights = load_data(directory,contains,f_ending,threshold,alpha_threshold,r_squared_threshold)

plt.figure(figsize=(10, 6))

cut_index = 0
dataframe_wav = np.round(np.arange(910, 980 + 0.001, 0.1), 1)

file = '850_AlGaAs_1300_scattering_7p0_0p6angle_0_fund_higher2350_.hdf'

f = klepto.archives.file_archive(file)
f.load()

wg_width = np.array(f["wg_width"])
wg_height = np.array(f["wg_height"])
wavelengths = np.array(f["wavelengths"])
sT = np.array(f["sT"])
sv = np.array(f["sv"])
sh = np.array(f["sh"])

sT_TM = sT[0]
sT_TE = sT[1]
sv_TM = sv[0]
sv_TE = sv[1]
sh_TM = sh[0]
sh_TE = sh[1]

idx = wavelengths > 890
wavelengths = wavelengths[idx]
sT_TM = sT_TM[idx]
sT_TE = sT_TE[idx]
sv_TM = sv_TM[idx]
sv_TE = sv_TE[idx]
sh_TM = sh_TM[idx]
sh_TE = sh_TE[idx]

plot_fontsize = 21
te_dataframe, te_mean, te_std = plot_polarization_data(dataframe_wav, TE_data, TE_wav, "TE", "b",TE_weights)
tm_dataframe, tm_mean, tm_std = plot_polarization_data(dataframe_wav, TM_data, TM_wav, "TM", "r",TM_weights)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.figure(figsize=(10, 7))

c1 = "#377eb8"
c2 = "#a65628"
c3 = "#000000"

if not te_dataframe.empty:
    plt.plot(te_mean.index, te_mean,color=c1,label="TE")
    plt.fill_between(te_mean.index, te_mean - te_std, te_mean + te_std, alpha=0.2,color=c1)
    plt.plot(te_mean.index, te_mean + te_std,linestyle="-", color=c1, alpha=0.3)
    plt.plot(te_mean.index, te_mean - te_std,linestyle="-", color=c1, alpha=0.3)

    plt.xlabel("Wavelength (nm)",fontsize = plot_fontsize)
    plt.ylabel("Alpha (dB/cm)",fontsize = plot_fontsize)
    plt.legend(fontsize=plot_fontsize)
    plt.xlim(910,980)
    plt.ylim(0,75)
    plt.xticks(fontsize=plot_fontsize)
    plt.yticks(fontsize=plot_fontsize)

if not tm_dataframe.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(tm_mean.index, tm_mean,color=c1,label="TM")
    plt.fill_between(tm_mean.index, tm_mean - tm_std, tm_mean + tm_std, alpha=0.2,color=c1)
    plt.plot(tm_mean.index, tm_mean + tm_std, linestyle="-", color=c1, alpha=0.2)
    plt.plot(tm_mean.index, tm_mean - tm_std, linestyle="-", color=c1, alpha=0.2)

    plt.xlabel("Wavelength (nm)",fontsize=plot_fontsize)
    plt.ylabel("Alpha (dB/cm)",fontsize=plot_fontsize)

    plt.legend(fontsize=plot_fontsize)
    plt.xticks(fontsize=plot_fontsize)
    plt.yticks(fontsize=plot_fontsize)
    plt.xlim(910,980)

# %%
#if not te_dataframe.empty:
#    hf = h5py.File(directory + contains + "_TE.h5", "w")
#    hf.create_dataset("wavelengths", data=te_mean.index)
#    hf.create_dataset("Average TE loss (dB per cm)", data=te_mean.iloc[cut_index:])
#    hf.create_dataset("Upper confidencebound", data=te_mean + te_std)
#    hf.create_dataset("Lower confidencebound", data=te_mean - te_std)
#    hf.close()
#if not tm_dataframe.empty:
#    hf = h5py.File(directory + contains + "_TM.h5", "w")
#    hf.create_dataset("wavelengths", data=tm_mean.index)
#    hf.create_dataset("Average TM loss (dB per cm)", data=tm_mean.iloc[cut_index:])
#    hf.create_dataset("Upper confidencebound", data=tm_mean + tm_std)
#    hf.create_dataset("Lower confidencebound", data=tm_mean - tm_std)
#    hf.close()

plt.show()
