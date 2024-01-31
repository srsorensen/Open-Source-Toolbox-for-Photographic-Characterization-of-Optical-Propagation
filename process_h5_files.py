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


def plot_polarization_data(wavelengths, data, data_wav, polarization_label, color, cutlist, weights):
    if len(data) > 0:
        dataframe = pd.DataFrame(columns=wavelengths)
        dataframe_weights = pd.DataFrame(columns=wavelengths)

        for i in range(len(data)):
            series = pd.Series(data[i], index=data_wav[i])
            dataframe = pd.concat([dataframe, series.to_frame().T], ignore_index=True)

        for i in range(len(weights)):
            series = pd.Series(weights[i], index=data_wav[i])
            dataframe_weights = pd.concat([dataframe_weights, series.to_frame().T], ignore_index=True)
        print(dataframe_weights)
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
        # mean = dataframe.mean()
        # plt.plot(mean.index, mean, f"{color}-.", label=f"{polarization_label} mean")
        return dataframe, weighted_average_sweep, weighted_std_sweep
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


directory = "E:/Top_Down_Method/"
contains = "980nm_optimized_ST3_width_1600nm_TE_1_100"
f_ending = ".h5"

polarization_plot_dict = {"TM": "b-", "TE": "r-"}

TE_data = []
TM_data = []

TE_weights = []
TM_weights = []

TE_wav = []
TM_wav = []

TE_dict = {}
TM_dict = {}

for file in os.listdir(directory):
    if contains.lower() in file.lower() and f_ending.lower() in file.lower():
        print(file)
        hf = h5py.File(directory + file, 'r')
        wav_nm = np.array(hf.get('wavelength'))
        weights = np.array(hf.get('alpha_variance'))
        # print(weights)
        # print(wav_nm)
        alphas = np.array(hf.get('alpha'))

        y_savgol = savgol_filter(alphas.tolist(), 501, 1, mode="nearest")
        # polarization = get_polarization_name(file)
        if "TE" in file:
            TE_data.append(list(y_savgol))
            TE_weights.append(weights)
            TE_wav.append(([float(x.decode()) for x in wav_nm]))
        else:
            TM_data.append(list(y_savgol))
            TM_weights.append(weights)
            TM_wav.append(([float(x.decode()) for x in wav_nm]))
        hf.close()

directory = "E:/Top_Down_Method/"
contains = "980nm_optimized_ST3_width_1600nm_TE_1_200"
f_ending = ".h5"


polarization_plot_dict = {"TM": "b-", "TE": "r-"}

TE_data1 = []
TM_data1 = []
TE_wav1 = []
TM_wav1 = []

TE_weights1 = []
TM_weights1 = []

TE_dict = {}
TM_dict = {}

for file in os.listdir(directory):
    if contains.lower() in file.lower() and f_ending.lower() in file.lower():
        #print(file)
        hf = h5py.File(directory + file, 'r')
        wav_nm = np.array(hf.get('wavelength'))
        weights = np.array(hf.get('alpha_variance'))
        # print(weights)
        # print(wav_nm)
        alphas = np.array(hf.get('alpha'))
        y_savgol = savgol_filter(alphas.tolist(), 501, 1, mode="nearest")
        # polarization = get_polarization_name(file)
        if "TE" in file:
            TE_data1.append(list(y_savgol))
            TE_weights1.append(weights)
            TE_wav1.append(([float(x.decode()) for x in wav_nm]))
        else:
            TM_data1.append(list(y_savgol))
            TM_weights1.append(weights)
            TM_wav1.append(([float(x.decode()) for x in wav_nm]))
        hf.close()

directory = "E:/Top_Down_Method/"
contains = "980nm_optimized_ST3_width_1600nm_TE_1_300"
f_ending = ".h5"


polarization_plot_dict = {"TM": "b-", "TE": "r-"}

TE_data2 = []
TM_data2 = []
TE_wav2 = []
TM_wav2 = []

TE_weights2 = []
TM_weights2 = []

TE_dict = {}
TM_dict = {}

for file in os.listdir(directory):
    if contains.lower() in file.lower() and f_ending.lower() in file.lower():
        #print(file)
        hf = h5py.File(directory + file, 'r')
        wav_nm = np.array(hf.get('wavelength'))
        weights = np.array(hf.get('alpha_variance'))
        # print(weights)
        # print(wav_nm)
        alphas = np.array(hf.get('alpha'))
        y_savgol = savgol_filter(alphas.tolist(), 501, 1, mode="nearest")
        # polarization = get_polarization_name(file)
        if "TE" in file:
            TE_data2.append(list(y_savgol))
            TE_weights2.append(weights)
            TE_wav2.append(([float(x.decode()) for x in wav_nm]))
        else:
            TM_data2.append(list(y_savgol))
            TM_weights2.append(weights)
            TM_wav2.append(([float(x.decode()) for x in wav_nm]))
        hf.close()

directory = "E:/Top_Down_Method/"
contains = "980nm_optimized_ST3_width_1600nm_TE_1_opt"
f_ending = ".h5"


polarization_plot_dict = {"TM": "b-", "TE": "r-"}

TE_data3 = []
TM_data3 = []
TE_wav3 = []
TM_wav3 = []

TE_weights3 = []
TM_weights3 = []

TE_dict = {}
TM_dict = {}

for file in os.listdir(directory):
    if contains.lower() in file.lower() and f_ending.lower() in file.lower():
        #print(file)
        hf = h5py.File(directory + file, 'r')
        wav_nm = np.array(hf.get('wavelength'))
        weights = np.array(hf.get('alpha_variance'))
        # print(weights)
        # print(wav_nm)
        alphas = np.array(hf.get('alpha'))
        y_savgol = savgol_filter(alphas.tolist(), 501, 1, mode="nearest")
        # polarization = get_polarization_name(file)
        if "TE" in file:
            TE_data3.append(list(y_savgol))
            TE_weights3.append(weights)
            TE_wav3.append(([float(x.decode()) for x in wav_nm]))
        else:
            TM_data3.append(list(y_savgol))
            TM_weights3.append(weights)
            TM_wav3.append(([float(x.decode()) for x in wav_nm]))
        hf.close()

plt.figure(figsize=(10, 6))

cut_index = 0
dataframe_wav = np.round(np.arange(910, 980 + 0.001, 0.1), 1)

# TM_data[2] = TM_data[2][:451]
# TM_wav[2] = TM_wav[2][:451]

# print([len(x) for x in TM_data20mW])

te_dataframe, te_mean, te_std = plot_polarization_data(dataframe_wav, TE_data, TE_wav, "TE", "b", [910, 980.1],TE_weights)
te_dataframe1, te_mean1, te_std1 = plot_polarization_data(dataframe_wav,TE_data1,TE_wav1,"TE","b",[910,980.1],TE_weights1)
te_dataframe2, te_mean2, te_std2 = plot_polarization_data(dataframe_wav,TE_data2,TE_wav2,"TE","b",[910,980.1],TE_weights2)
te_dataframe3, te_mean3, te_std3 = plot_polarization_data(dataframe_wav,TE_data3,TE_wav3,"TE","b",[910,980.1],TE_weights3)
tm_dataframe, tm_mean, tm_std = plot_polarization_data(dataframe_wav, TM_data, TM_wav, "TM", "r", [910, 980.1],TM_weights)
tm_dataframe1, tm_mean1, tm_std1 = plot_polarization_data(dataframe_wav,TM_data1,TM_wav1,"TM","r",[910,980.1],TM_weights1)
tm_dataframe2, tm_mean2, tm_std2 = plot_polarization_data(dataframe_wav,TM_data2,TM_wav2,"TM","b",[910,980.1],TM_weights2)
tm_dataframe3, tm_mean3, tm_std3 = plot_polarization_data(dataframe_wav,TM_data3,TM_wav3,"TM","b",[910,980.1],TM_weights3)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.figure(figsize=(10, 6))

te_max = te_dataframe.idxmax(axis=1)
tm_max = tm_dataframe.idxmax(axis=1)

if not te_dataframe.empty:
    plt.plot(te_mean.index, te_mean, label="100 left indent")
    plt.fill_between(te_mean.index, te_mean - te_std, te_mean + te_std, alpha=0.2, label="Confidence band")
    plt.plot(te_mean1.index, te_mean1, label="200 left indent")
    plt.fill_between(te_mean1.index, te_mean1 - te_std1, te_mean1 + te_std1, alpha=0.2, label="Confidence band")
    plt.plot(te_mean2.index, te_mean2, label="300 left indent")
    plt.fill_between(te_mean2.index, te_mean2 - te_std2, te_mean2 + te_std2, alpha=0.2,label="Confidence band")
    plt.plot(te_mean3.index, te_mean3, label="Optimized left indent")
    plt.fill_between(te_mean3.index, te_mean3 - te_std3, te_mean3 + te_std3, alpha=0.2)
    #plt.axvline(max(te_max), color='r', linestyle='--', label='Mean: ' + str(round(max(te_max), 1)) + 'nm')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Alpha (dB/cm)")
    plt.legend()
    # plt.xlim([938,980])

if not tm_dataframe.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(tm_mean.index, tm_mean, label="100 left indent")
    plt.fill_between(tm_mean.index, tm_mean - tm_std, tm_mean + tm_std, alpha=0.2)
    plt.plot(tm_mean1.index, tm_mean1, label="200 left indent")
    plt.fill_between(tm_mean1.index, tm_mean1 - tm_std1, tm_mean1 + tm_std1, alpha=0.2)
    plt.plot(tm_mean2.index, tm_mean2, label="300 left indent")
    plt.fill_between(tm_mean2.index, tm_mean2 - tm_std2, tm_mean2 + tm_std2, alpha=0.2)
    plt.plot(tm_mean3.index, tm_mean3, label="Optimized left indent")
    plt.fill_between(tm_mean3.index, tm_mean3 - tm_std3, tm_mean3 + tm_std3, alpha=0.2)
    #plt.axvline(max(tm_max), color='r', linestyle='--', label='Mean: ' + str(round(max(tm_max), 1)) + 'nm')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Alpha (dB/cm)")
    plt.legend()

# %%
if not te_dataframe.empty:
    hf = h5py.File("E:/Average/" + contains + "_TE.h5", "w")
    hf.create_dataset("wavelengths", data=te_mean.index)
    hf.create_dataset("Average TE loss (dB per cm)", data=te_mean.iloc[cut_index:])
    hf.create_dataset("Upper confidencebound", data=te_mean + te_std)
    hf.create_dataset("Lower confidencebound", data=te_mean - te_std)
    hf.close()
if not tm_dataframe.empty:
    hf = h5py.File("E:/Average/" + contains + "_TM.h5", "w")
    hf.create_dataset("wavelengths", data=tm_mean.index)
    hf.create_dataset("Average TM loss (dB per cm)", data=tm_mean.iloc[cut_index:])
    hf.create_dataset("Upper confidencebound", data=tm_mean + tm_std)
    hf.create_dataset("Lower confidencebound", data=tm_mean - tm_std)
    hf.close()

plt.show()