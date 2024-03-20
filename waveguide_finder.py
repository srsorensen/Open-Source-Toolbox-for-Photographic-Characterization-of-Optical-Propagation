# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:35:46 2023

@author: frede
"""
#Click and find input/output on picture.

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
from scipy.signal import find_peaks, savgol_filter, convolve, convolve2d


def mean_image_intensity(image, disk_size, q1=0, q3=1):
    mean_disk = disk(disk_size)

    mean_image = (rank.mean_percentile(image, footprint=mean_disk, p0=q1, p1=q3))
    return mean_image


def find_path(bw_image, start, end):
    costs = np.where(bw_image == 1, 1, 10000)
    path, cost = skimage.graph.route_through_array(costs, start=start, end=end, fully_connected=True, geometric=True)
    return path, cost



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

from math import isclose
def opt_indent(parameter,x_iqr,y_iqr,y_savgol):
    if parameter == "left indent":
        indent = np.arange(0, 800, 20)
        dI = indent[1] - indent[0]
        alpha_dB_i = []
        for i in range(len(indent)):  # 525
            x_iqr = x_iqr[i:]
            y_iqr = y_iqr[i:]
            y_savgol = y_savgol[i:]
            fit_x = x_iqr
            intensity_values = y_savgol
            initial_guess = [25, 0.0006, np.min(intensity_values)]
            fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(exponential_function_offset,
                                                                                            fit_x,
                                                                                            intensity_values,
                                                                                            p0=initial_guess,
                                                                                            full_output=True,
                                                                                            maxfev=5000)  # sigma=weights, absolute_sigma=True
            fit = exponential_function_offset(fit_x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
            fit_guess = exponential_function_offset(fit_x, *initial_guess)
            residuals = fit - intensity_values
            mean_squared_error = np.mean(residuals ** 2)

            residuals = intensity_values - exponential_function_offset(fit_x, *fit_parameters)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((intensity_values - np.mean(intensity_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 10))
            alpha_dB_i.append(alpha_dB)
        smoothed_alpha = savgol_filter(alpha_dB_i, 4, 1,mode='nearest')
        alpha_indent = np.gradient(smoothed_alpha, dI)
        index_min = []
        abs_tol = 0.01
        while abs_tol < 0.9:
            if len(index_min) == 0:
                zero_gradient_minus = []
                zero_gradient_plus = []
                for i in range(len(indent) - 1):
                    zero_gradient_p = isclose(alpha_indent[i], alpha_indent[i + 1], abs_tol=abs_tol)
                    zero_gradient_m = isclose(alpha_indent[i], alpha_indent[i - 1], abs_tol=abs_tol)
                    zero_gradient_minus.append(zero_gradient_m)
                    zero_gradient_plus.append(zero_gradient_p)
                    if zero_gradient_plus[i] == True and zero_gradient_minus[i] == True:
                        index_min.append(i)
            abs_tol = abs_tol + 0.01
        num_neighbors = 1
        point_mean = []
        for index in index_min:
            neighbor_indexes = np.arange(index - num_neighbors, index + num_neighbors + 1, 1)
            neighbor_indexes = [x for x in neighbor_indexes if x > 0 and x < len(alpha_indent)]
            point_m = np.mean(smoothed_alpha[neighbor_indexes])
            point_mean.append(point_m)
        absolute_point_mean = [abs(num) for num in point_mean]
        min_point_mean = point_mean.index(min(absolute_point_mean))
        ideal_indent = indent[index_min[min_point_mean]]
        plt.figure(figsize=(10, 6))
        plt.plot(indent, alpha_indent, "k")
        plt.xlabel("Left indents", fontsize=font_size)
        plt.ylabel("d$d\\alpha$/d(indent)", fontsize=font_size)
        plt.axvline(ideal_indent, color="r", linestyle="--")
        plt.show()
    return ideal_indent


path = "C:/Users/simon/PycharmProjects/Open-Source-Toolbox-for-Rapid-and-Accurate-Photographic-Characterization-of-Optical-Propagation/2023-09-08_10_24_16_651_w31_1.3_waveguide2_spiral.png"

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

out_point = (out_point[0], out_point[1] + 210) #(1886,1208)
in_point = (in_point[0], in_point[1] - 15)  # -80

plt.figure(figsize=(10, 6))
plt.title("Grayscale Image with input and output and max pixel values removed")
plt.plot(*in_point, "ro")
plt.plot(*out_point, "ro")
imshow(grey_image)

point1 = np.array((in_point[0], in_point[1]))
point2 = np.array((in_point[0], out_point[1]))#1985

plt.plot(*point1, "bo")
plt.plot(*point2, "bo")

distance_um = 1.399#1102  # Measured in klayout
mum_per_pixel = um_per_pixel(point1, point2, distance_um)

sobel_h = ndi.sobel(grey_image, 0)
sobel_v = ndi.sobel(grey_image, 1)
magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
#magnitude_norm = magnitude / np.max(magnitude)
#indices_over_099 = np.argwhere(magnitude_norm > 0.04)
#x_coords = [coord[0] for coord in indices_over_099]
#y_coords = [coord[1] for coord in indices_over_099]

#plt.figure(figsize=(10,6))
#plt.scatter(y_coords,x_coords)
#plt.scatter(y_coords[0],x_coords[0])
#plt.ylim(2000,0)
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.title('Plot of Coordinates')
#plt.show()


path_length = []
threshold = np.round(np.linspace(0.01,0.1,10),2)
j = 0
for i in threshold:
    bw_waveguide = grey_image > i
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
    x_path.append(path_length[max_index][i][1])
    y_path.append(path_length[max_index][i][0])

font_size = 16

plt.figure()
# plt.title("Image with path")
plt.plot(*in_point, "ro")
plt.plot(*out_point, "ro")
plt.scatter(x_path[::100], y_path[::100], s=16, alpha=1, color="red")
plt.xlabel("Width [a.u.]", fontsize=font_size)
plt.ylabel("Height [a.u.]", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
# plt.axis('off')
plt.imshow(magnitude, cmap="turbo")

disk_size = 20
row = 20
col = 3
q1 = 0.00
q3 = 1

mean_image = mean_image_intensity(grey_image, disk_size)

y_raw = mean_image[y_path, x_path]

x_image = range(len(y_raw))

x = np.array([x * mum_per_pixel for x in x_image])


x_iqr, y_iqr, indexes = remove_outliers_IQR(x, y_raw, 10, 1)

y_savgol = savgol_filter(y_iqr, 2000, 1)

plt.figure(figsize=(10, 6))
plt.title("Mean of Image")
imshow(mean_image, cmap="twilight_shifted")

intensity_values = mean_image[y_path, x_path]

plt.figure()
# plt.title("Mean of intensity values as a function of distance, with background")
plt.scatter(x, y_raw, color="k", s=1, alpha=0.4)
plt.scatter(x_iqr, y_iqr, s=2, color="b", alpha=0.7)
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

l = opt_indent("left indent",x_iqr,y_iqr,y_savgol)
print(l)
#####################################3

x_iqr, y_iqr, indexes = remove_outliers_IQR(x, y_raw, 10, 1)

y_savgol = savgol_filter(y_iqr, 2000, 1)

indent = np.arange(1,801,20)
alpha_dB_i = []
alpha_dB_variance_i = []
for i in range(1,len(indent)):#525
    x_iqr = x_iqr[:-i]
    y_iqr = y_iqr[:-i]
    y_savgol = y_savgol[:-i]
    fit_x = x_iqr
    intensity_values = y_savgol
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

    alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 10))
    alpha_dB_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 10))
    alpha_dB_i.append(alpha_dB)
    alpha_dB_variance_i.append(alpha_dB_variance)
dI = indent[1] - indent[0]
alpha_dB_i = savgol_filter(alpha_dB_i,4,1,mode='nearest')
alpha_indent = np.gradient(alpha_dB_i, dI)
r = indent[np.argmax(alpha_indent)]
plt.figure(figsize=(10,6))
plt.plot(indent[1:len(indent)],alpha_indent,"k")
plt.xlabel("Right indents",fontsize=font_size)
plt.ylabel("d$d\\alpha$/d(indent)",fontsize=font_size)
plt.axvline(r,color="r",linestyle="--")
plt.show()
print(l)
print(r)
x_iqr, y_iqr, indexes = remove_outliers_IQR(x, y_raw, 10, 1)

y_savgol = savgol_filter(y_iqr, 2000, 1)


x_iqr = x_iqr[l:-r]
y_iqr = y_iqr[l:-r]
y_savgol = y_savgol[l:-r]
x = x[l:-r]
y_raw = y_raw[l:-r]

fit_x = x_iqr
intensity_values = y_savgol

initial_guess = [25, 0.0006, np.min(intensity_values)]
fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(exponential_function_offset, fit_x,
                                                                                intensity_values, p0=initial_guess,
                                                                                full_output=True,
                                                                                maxfev=5000)  # sigma=weights, absolute_sigma=True
fit = exponential_function_offset(fit_x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
fit_guess = exponential_function_offset(fit_x, *initial_guess)
residuals = fit - intensity_values
mean_squared_error = np.mean(residuals ** 2)

residuals = intensity_values - exponential_function_offset(fit_x, *fit_parameters)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((intensity_values - np.mean(intensity_values)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 10))
alpha_dB_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 10))

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

alpha_dB_raw = 10 * np.log10(np.exp(fit_parameters[1] * 10))
alpha_dB_raw_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 10))

initial_guess = [25, 0.0006, np.min(y_raw)]
fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(exponential_function_offset, x,
                                                                                y_raw, p0=initial_guess,
                                                                                full_output=True,maxfev=5000)  # sigma=weights, absolute_sigma=True
fit_r = exponential_function_offset(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
fit_guess = exponential_function_offset(x, *initial_guess)
residuals = fit_r - y_raw
mean_squared_error = np.mean(residuals ** 2)

residuals = y_raw - exponential_function_offset(x, *fit_parameters)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
r_squared_r = 1 - (ss_res / ss_tot)

alpha_dB_r = 10 * np.log10(np.exp(fit_parameters[1] * 10))
alpha_dB_r_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 10))

plt.figure()

plt.plot(x_iqr, fit_raw, color="#E69F00",linestyle="-", linewidth=3,label=f"Fit to outlier corrected data\n {alpha_dB_raw:.1f}$\pm${alpha_dB_raw_variance:.1f} dB/cm, R\u00b2: {r_squared_raw:.2f}")  # ,
plt.plot(x, fit_r, color="g",linestyle="-", linewidth=3,label=f"Fit to raw data\n {alpha_dB_r:.1f}$\pm${alpha_dB_r_variance:.1f} dB/cm, R\u00b2: {r_squared_r:.2f}")  # ,
plt.scatter(x, y_raw, color="#0072B2", s=1.5, label="Raw data")
plt.scatter(x_iqr, y_iqr, color="#000000", s=1.5,label="Outlier corrected data")
lgnd = plt.legend(fontsize=14, scatterpoints=1, frameon=False)
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[2].set_alpha(1)
lgnd.legendHandles[3]._sizes = [30]
lgnd.legendHandles[3].set_alpha(1)
plt.xlabel('Propagation length [mm]', fontsize=font_size)
plt.ylabel('Mean intensity [a.u.]', fontsize=font_size)
plt.xlim([min(x), max(x)])
plt.ylim([min(y_raw), max(y_raw)+5])
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
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
