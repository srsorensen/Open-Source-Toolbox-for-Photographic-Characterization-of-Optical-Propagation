# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:35:46 2023

@author: frede
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.graph
from skimage.io import imread, imshow
from skimage.morphology import disk, rectangle
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.filters import rank, gaussian
from skimage import util

from functions import *

import scipy.ndimage as ndi
from scipy.fft import ifft2, fftshift, fft2, ifftshift


def mean_image_intensity(image, disk_size, q1=0, q3=1):
    mean_disk = disk(disk_size)

    mean_image = (rank.mean_percentile(image, footprint=mean_disk, p0=q1, p1=q3))
    return mean_image


def mean_image_intensity_rectangular(image, row, col, q1=0, q3=1):
    mean_rectangle = rectangle(row, col)
    mean_image = (rank.mean_percentile(image, footprint=mean_rectangle, p0=q1, p1=q3))
    return mean_image


def find_path(bw_image, start, end):
    costs = np.where(bw_image == 1, 1, 10000)
    path, cost = skimage.graph.route_through_array(costs, start=start, end=end, fully_connected=True, geometric=True)
    return path, cost


def remove_background(image):  # FFT image and set center pixel to 0 to remove background and IFFT back to real image
    image_f = fftshift(fft2(image))
    shape = image.shape
    x = int(shape[0] / 2)
    y = int(shape[1] / 2)
    image_f[x, y] = 0

    background_f = fftshift(fft2(image)) - image_f

    background = abs(ifft2(ifftshift(background_f)))

    fft_img_mod = ifftshift(image_f)
    img_mod = abs(ifft2(fft_img_mod))
    return img_mod, background


def find_input_and_output(indent_list, image):
    input_indent_start = int(image.shape[1] * indent_list[0])
    input_indent_end = int(image.shape[1] * indent_list[1])

    output_indent_start = int(image.shape[1] * indent_list[2])
    output_indent_end = int(image.shape[1] * indent_list[3])

    input_index = image[:, input_indent_start:input_indent_end] > 0.02
    imshow(input_index)

    cy, cx = ndi.center_of_mass(input_index)

    cx = cx + input_indent_start

    input_point = (int(cx), int(cy))
    # print(input_point)

    output_index = image[:, output_indent_start:output_indent_end] > 0.02
    cy, cx = ndi.center_of_mass(output_index)
    cx = image.shape[1] - cx

    output_point = (int(cx), int(cy))

    return input_point, output_point


def um_per_pixel(point1, point2, distance):
    # calculating Euclidean distance
    dist_pixels = np.linalg.norm(point1 - point2)
    return distance / dist_pixels


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def remove_outliers_IQR(intensity):
    temp_intensity = intensity
    new_x = [range(len(temp_intensity))]

    Q1 = np.percentile(temp_intensity, 25, method='midpoint')
    Q3 = np.percentile(temp_intensity, 75, method='midpoint')
    IQR = Q3 - Q1

    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR

    print(upper)
    print(lower)

    upper_array = np.where(temp_intensity >= upper)[0]
    lower_array = np.where(temp_intensity <= lower)[0]

    remove_array = np.concatenate((upper_array, lower_array))

    new_intensity = np.delete(temp_intensity, remove_array)
    new_x = np.delete(new_x, remove_array)
    return new_x, new_intensity


def remove_outliers_IQR_2(x, data, blocks):
    data_blocks = np.array_split(data, blocks)
    x_blocks = np.array_split(x, blocks)

    for i in range(len(data_blocks)):
        Q1 = np.percentile(data_blocks[i], 25, method='midpoint')
        Q3 = np.percentile(data_blocks[i], 75, method='midpoint')
        IQR = Q3 - Q1

        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR

        upper_array = np.where(data_blocks[i] >= upper)[0]
        lower_array = np.where(data_blocks[i] <= lower)[0]

        remove_array = np.concatenate((upper_array, lower_array))

        data_blocks[i] = np.delete(data_blocks[i], remove_array)
        x_blocks[i] = np.delete(x_blocks[i], remove_array)

    return np.concatenate(x_blocks), np.concatenate(data_blocks)


def EMA(data, smoothing=0.01):
    i = 1
    ma = []

    ma.append(data[0])

    while i < len(data):
        window_average = smoothing * data[i] + (1 - smoothing) * ma[-1]
        ma.append(window_average)
        i += 1
    return ma

path = "C:/Users/simon/PycharmProjects/SPA/Data/2023-09-07_16_10_03_312_chip13_waveguide3.png"
# path2 = "C:/Users/frede/OneDrive/Skrivebord/Civil/Speciale/2023-09-08_10_24_16_651_w31_1.3_waveguide2_spiral.png"
# path3 = "C:/Users/frede/OneDrive/Skrivebord/Civil/Speciale/W31 1.4 air measurements/08-08-23/2023-09-07_16_10_03_312_chip14_waveguide4.png"


image = util.img_as_float(imread(path))
# image = (rotate(image,180,resize=True))

plt.figure(figsize=(10, 6))
plt.title("Histogram of channels")
plt.hist(image[:, :, 2].ravel(), bins=256, histtype='step', color='blue')
plt.hist(image[:, :, 1].ravel(), bins=256, histtype='step', color='green')
plt.hist(image[:, :, 0].ravel(), bins=256, histtype='step', color='red')
plt.yscale("log")

plt.figure(figsize=(10, 6))
plt.title("Original Image")
imshow(image)

# grey_image = rgb2gray(image)
grey_image = image[:, :, 2]

indent_list = [0, 0.05, 0.9, 1]
in_point, out_point = find_input_and_output(indent_list, grey_image)

in_point, out_point
#1886,1208
out_point = (1886,1208)#(out_point[0], out_point[1] + 210)
in_point = (in_point[0], in_point[1] - 15)  # -80

plt.figure(figsize=(10, 6))
plt.title("Grayscale Image with input and output and max pixel values removed")
plt.plot(*in_point, "ro")
plt.plot(*out_point, "ro")

point1 = np.array((in_point[0], in_point[1]))
point2 = np.array((in_point[0], 1985))#out_point[1]

plt.plot(*point1, "bo")
plt.plot(*point2, "bo")

distance_um = 1399#1102  # Measured in klayout
mum_per_pixel = um_per_pixel(point1, point2, distance_um)

# peaks = grey_image != 1
# grey_image = grey_image*peaks
imshow(grey_image)

# grey_image, background = remove_background(grey_image)
sobel_h = ndi.sobel(grey_image, 0)  # horizontal gradient
sobel_v = ndi.sobel(grey_image, 1)  # vertical gradient
magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
magnitude *= 255.0 / np.max(magnitude)  # normalization

# %%
from scipy.signal import savgol_filter


def remove_outliers_IQR(x, data, blocks, num_neighbors):
    # Removal of outliers using IQR. Change blocks -> Num_Subsets
    data_blocks = np.array_split(data, blocks)
    x_blocks = np.array_split(x, blocks)
    x_blocks_indexes = [x[-1] for x in x_blocks]

    for i in range(len(data_blocks)):
        Q1 = np.percentile(data_blocks[i], 25, interpolation='midpoint')
        Q3 = np.percentile(data_blocks[i], 75, interpolation='midpoint')
        IQR = Q3 - Q1

        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR

        upper_array = np.where(data_blocks[i] >= upper)[0]
        lower_array = np.where(data_blocks[i] <= lower)[0]

        remove_array = np.concatenate((upper_array, lower_array))
        new_remove_array = []

        for index in remove_array:  # Finding indexes of neighbors to detected outliers
            neighbor_indexes = np.arange(index - num_neighbors, index + num_neighbors + 1, 1)
            neighbor_indexes = [x for x in neighbor_indexes if x > 0 and x < len(data_blocks[i])]
            new_remove_array += neighbor_indexes

        new_remove_array = list(set(new_remove_array))
        data_blocks[i] = np.delete(data_blocks[i], new_remove_array)  # removing outliers and neighbors from data
        x_blocks[i] = np.delete(x_blocks[i], new_remove_array)

    return np.concatenate(x_blocks), np.concatenate(data_blocks), x_blocks_indexes


font_size = 13
path_length = []
threshold = np.round(np.linspace(0.01,0.1,10),2)
j = 0
for i in threshold:
    bw_waveguide = grey_image > i
    #plt.figure(figsize=(10, 6))
    #imshow(bw_waveguide)
    #plt.axis('off')
    start = (in_point[1], in_point[0])
    end = (out_point[1], out_point[0])
    path, costs = find_path(bw_waveguide, start, end)
    path_length.append(path)

diff_paths = []
path_length_mum = []
for element in path_length:
    sub_length = len(element)
    length_mum = sub_length*mum_per_pixel
    diff_paths.append(sub_length)
    path_length_mum.append(length_mum)

max_element = max(path_length_mum)
max_index = path_length_mum.index(max_element)

x_path = []
y_path = []
for i in range(len(path_length[max_index])):
    # if i % 10 == 0:
    x_path.append(path_length[max_index][i][1])
    y_path.append(path_length[max_index][i][0])
#print(len(path)*mum_per_pixel)
# x_path = x_path[1050:-1200]
# y_path = y_path[1050:-1200]

plt.figure()
# plt.title("Image with path")
plt.plot(*in_point, "ro")
plt.plot(*out_point, "ro")
plt.scatter(x_path[::100], y_path[::100], s=16, alpha=1, color="red")
plt.xlabel("Width in pixels", fontsize=font_size)
plt.ylabel("Height in pixels", fontsize=font_size)
# plt.axis('off')
plt.imshow(magnitude, cmap="turbo")

disk_size = 20
row = 20
col = 3
q1 = 0.00
q3 = 1

mean_image = mean_image_intensity(grey_image, disk_size)
# mean_image = mean_image_intensity_rectangular(grey_image, row, col,q1,q3)

# mean_test = mean_image_intensity_rectangular(grey_image, row, col)

y_raw = mean_image[y_path, x_path]

x_image = range(len(y_raw))

x = np.array([x * mum_per_pixel for x in x_image])
# test_x, new_intensity = remove_outliers_IQR()

x_iqr, y_iqr, indexes = remove_outliers_IQR(x, y_raw, 10, 5)

y_savgol = savgol_filter(y_iqr, 2000, 1)

plt.figure(figsize=(10, 6))
plt.title("Mean of Image")
imshow(mean_image, cmap="twilight_shifted")

intensity_values = mean_image[y_path, x_path]
# intensity_test = mean_test[y_path,x_path]

# background_intensity = background[y_path,x_path]


# mov_intensity_values = moving_average(new_intensity, 100)

plt.figure()
# plt.title("Mean of intensity values as a function of distance, with background")
plt.scatter(x, y_raw, color="k", s=1, alpha=0.4)

plt.scatter(x_iqr, y_iqr, s=1, color="b", alpha=0.7)
# plt.plot(x_exp,y_exp,"g-")
plt.plot(x_iqr, y_savgol, "r-", linewidth=3)
# plt.plot(background_intensity)
plt.xlabel('x Length [um]', fontsize=font_size)
plt.ylabel('Mean pixel intensity [a.u.]', fontsize=font_size)
lgnd = plt.legend(["Raw data", "Outlier corrected", "Smoothed data"], fontsize=font_size, scatterpoints=1,
                  frameon=False)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[0].set_alpha(1)
lgnd.legendHandles[1].set_alpha(1)
lgnd.legendHandles[1]._sizes = [30]
plt.xlim([0, 8000])
plt.ylim([0, 110])
plt.show()

# intensity_values = new_intensity


# point1 = np.array((0,0))
# point2 = np.array((0,grey_image.shape[1]))
x_iqr = x_iqr[2150:5750]
y_iqr = y_iqr[2150:5750]
y_savgol = y_savgol[2150:5750]

fit_x = x_iqr
intensity_values = y_savgol

# fit_x = np.array([x*mum_per_pixel for x in fit_x])

initial_guess = [25, 0.0006, np.min(intensity_values)]
fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(exponential_function_offset, fit_x,
                                                                                intensity_values, p0=initial_guess,
                                                                                full_output=True,maxfev=5000)  # sigma=weights, absolute_sigma=True
fit = exponential_function_offset(fit_x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
fit_guess = exponential_function_offset(fit_x, *initial_guess)
residuals = fit - intensity_values
mean_squared_error = np.mean(residuals ** 2)

residuals = intensity_values - exponential_function_offset(fit_x, *fit_parameters)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((intensity_values - np.mean(intensity_values)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 1e4))
alpha_dB_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4))
# intensity_values = np.delete(intensity_values,remove_array)
# fit_x_new = np.delete(fit_x,remove_array)


# fit_parameters, fit_parameters_cov_var_matrix, infodict,mesg, ier,  = curve_fit(exponential_function, fit_x_new, intensity_values, p0=initial_guess, full_output=True)
# fit_new = exponential_function(fit_x_new, *fit_parameters)
# confidence_bounds_sigma_f = exponential_function_confidence_bounds_sigma(fit_x, fit_parameters[0], fit_parameters[1], fit_parameters_cov_var_matrix)
# prediction_bounds_sigma_f = exponential_function_prediction_bounds_sigma(fit_x, fit_parameters[0], fit_parameters[1], fit_parameters_cov_var_matrix, mean_squared_error)

plt.figure(figsize=(10, 6))
plt.yscale('log')
plt.plot(fit_x, intensity_values, 'b-', label="Smoothed data")
plt.plot(fit_x, fit, 'r-', label=f"Fit to smoothed data: {alpha_dB:.1f}$\pm${alpha_dB_variance:.1f} dB/cm, R\u00b2: {r_squared:.2f}")  # ,
plt.scatter(fit_x, y_iqr, alpha=0.1, label="Raw data", s=2, color="k")

lgnd = plt.legend(fontsize=font_size, scatterpoints=1, frameon=False)
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[2].set_alpha(1)
plt.xlabel('x Length [um]', fontsize=font_size)

plt.ylabel('Mean of blue intensity', fontsize=font_size)
plt.show()

initial_guess = [25, 0.0006, np.min(y_iqr)]
fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(exponential_function_offset, x_iqr,
                                                                                y_iqr, p0=initial_guess,
                                                                                full_output=True,maxfev=5000)  # sigma=weights, absolute_sigma=True
fit_raw = exponential_function_offset(x_iqr, fit_parameters[0], fit_parameters[1], fit_parameters[2])
fit_guess = exponential_function_offset(fit_x, *initial_guess)
residuals = fit - intensity_values
mean_squared_error = np.mean(residuals ** 2)

residuals = y_iqr - exponential_function_offset(fit_x, *fit_parameters)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_iqr - np.mean(y_iqr)) ** 2)
r_squared_raw = 1 - (ss_res / ss_tot)

alpha_dB_raw = 10 * np.log10(np.exp(fit_parameters[1] * 1e4))
alpha_dB_raw_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4))

plt.figure()

# plt.plot(fit_x, fit, 'r-',linewidth=3, label=f"Fit to smoothed data: {alpha_dB:.1f}$\pm${alpha_dB_variance:.1f} dB/cm") #,
plt.yscale('log')
plt.plot(x_iqr, fit_raw, 'b-', linewidth=3,
         label=f"Fit to outlier corrected data\n {alpha_dB_raw:.1f}$\pm${alpha_dB_raw_variance:.1f} dB/cm, R\u00b2: {r_squared_raw:.2f}")  # ,
plt.plot(fit_x, intensity_values, 'r--', label="Smoothed data", linewidth=2)
plt.scatter(fit_x, y_iqr, alpha=0.1, label="Raw data", s=2.5, color="k")

lgnd = plt.legend(fontsize=font_size, scatterpoints=1, frameon=False)
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[2].set_alpha(1)
plt.xlabel('x Length [um]', fontsize=font_size)

plt.ylabel('Mean pixel intensity [a.u.]', fontsize=font_size)
#plt.xlim([0, 8000])
#plt.ylim([0, 70])
plt.show()
print("Fit Parameters:", fit_parameters)
print("Variance-Covariance Matrix Fit Parameters:", fit_parameters_cov_var_matrix)
print(
    f'a={fit_parameters[0]} +- {np.sqrt(fit_parameters_cov_var_matrix[0, 0])}, b={fit_parameters[1]} +- {np.sqrt(fit_parameters_cov_var_matrix[1, 1])}')
print(f"alpha = {fit_parameters[1] * 1e4} +- {np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4} 1/cm")
print(
    f'alpha_dB = {10 * np.log10(np.exp(fit_parameters[1] * 1e4))} +- {10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4))}) dB/cm')
print(f'Length over measurement = {fit_x[-1] - fit_x[0]} um')

from sklearn.metrics import r2_score

coefficient_of_determination = r2_score(y_iqr, fit_raw)

print(f"R^2: {coefficient_of_determination}")


# %%
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

# %%
