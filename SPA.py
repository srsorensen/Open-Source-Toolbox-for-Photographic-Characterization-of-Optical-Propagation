# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:08:25 2023

@author: Peter Tønning, Kevin Bach Gravesen, Magnus Linnet Madsen, Frederik P
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, savgol_filter, convolve, convolve2d
from scipy.fft import ifft2, fftshift, fft2, ifftshift
from math import isclose


class SPA:
    def __init__(self, show_plots, chiplength, manual=False):
        self.show_plots = show_plots
        self.chiplength = chiplength  # 7229 um measured on the GDS, 2445 is the pixel width of the sensor (Both numbers inherent of the sensor and lens)
        self.manual = manual  # When true remember to call manual_input_output and set_um_per_pixel before analyze_image

    def set_um_per_pixel(self, point1, point2):
        # For straight waveguides: calculating Euclidean distance between input and output.
        points = [np.array(point1), np.array(point2)]
        dist_pixels = np.linalg.norm(points[0] - points[1])

        self.mum_per_pixel = self.chiplength / dist_pixels

    def get_intensity_array(self, image_array):
        return np.clip(
            np.sqrt(image_array[:, :, 0] ** 2 + image_array[:, :, 1] ** 2 + image_array[:, :, 2] ** 2) / np.sqrt(
                3 * 255 ** 2) * 255, 0, 255)

    def insertion_detection(self, image, show_plots=True):
        # Convert the image to a NumPy array
        image_array = np.array(image)
        image_array_shape = np.shape(image_array)

        scale_factor = 2 ** 3
        new_height = int(np.rint(image_array_shape[0] / scale_factor))  # New width in pixels
        new_width = int(np.rint(image_array_shape[1] / scale_factor))  # New height in pixels
        # print(new_height)
        # print(new_width)
        # Resize the image to the new dimensions
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        resized_image_array = np.array(resized_image)

        # Define a simple convolution kernel
        kernel_size = 4
        kernel = np.ones([kernel_size, kernel_size]) / kernel_size ** 2

        # Apply convolution separately for each color channel
        filtered_channels = []
        for channel in range(3):  # RGB channels
            filtered_channel = convolve2d(resized_image_array[:, :, channel], kernel, mode='same', boundary='wrap')
            filtered_channels.append(filtered_channel)

        # Combine the filtered channels back into an RGB image
        filtered_image_array = np.stack(filtered_channels, axis=2)

        intensity_filtered_image_array = self.get_intensity_array(filtered_image_array)
        # if show_plots:
        # plt.figure()
        # plt.imshow(intensity_filtered_image_array, cmap="jet", vmin=0, vmax=255)

        resized_input_index = np.unravel_index(np.argmax(intensity_filtered_image_array, axis=None),
                                               intensity_filtered_image_array.shape)

        input_height_index = int(resized_input_index[0] * scale_factor)
        input_width_index = int(resized_input_index[1] * scale_factor)

        height_tolerance_output = 100
        left_index_limit = int(new_width) - 20
        right_index_limit = int(new_width) - 1
        lower_index_limit = int((input_height_index - height_tolerance_output) / scale_factor)
        upper_index_limit = int((input_height_index + height_tolerance_output) / scale_factor)
        # if show_plots:
        # plt.plot([0, new_width], [upper_index_limit, upper_index_limit], 'r-')
        # plt.plot([0, new_width], [lower_index_limit, lower_index_limit], 'r-')
        # plt.plot([left_index_limit, left_index_limit], [lower_index_limit, upper_index_limit], 'r-')
        # plt.plot([right_index_limit, right_index_limit], [lower_index_limit, upper_index_limit], 'r-')
        # plt.plot([0, 0], [0, kernel_size], 'b-')
        # plt.plot([kernel_size, kernel_size], [0, kernel_size], 'b-')
        # plt.plot([0, kernel_size], [0, 0], 'b-')
        # plt.plot([0, kernel_size], [kernel_size, kernel_size], 'b-')
        output_array = intensity_filtered_image_array[lower_index_limit: upper_index_limit,
                       left_index_limit: right_index_limit]
        resized_output_index = np.unravel_index(np.argmax(output_array, axis=None), output_array.shape)

        output_height_index = int((resized_output_index[0] + lower_index_limit) * scale_factor)
        output_width_index = int((resized_output_index[1] + left_index_limit) * scale_factor)
        # if show_plots:
        # plt.plot(resized_input_index[1], resized_input_index[0], 'r.')
        # plt.plot(resized_output_index[1] + left_index_limit, resized_output_index[0] + lower_index_limit, 'r.')
        return input_width_index, input_height_index, output_width_index, output_height_index

    def find_waveguide_angle(self, image_array, left_index_guess, left_right_separation, number_of_points,
                             show_plots=True):
        kernel = np.ones([1, 50]) / (1 * 50)
        smoothed_image_array = convolve2d(image_array, kernel)
        # if show_plots:
        # plt.figure()
        # plt.imshow(smoothed_image_array)
        x_index_array = []
        max_height_index_array = []
        for index in range(0, number_of_points):
            x_index = left_index_guess + index * left_right_separation
            x_index_array.append(x_index)
            max_array = np.flip(np.mean(smoothed_image_array[:, x_index: left_index_guess + x_index + 1], axis=1))
            max_height_index = np.argmax(max_array - np.mean(max_array))
            max_height_index_array.append(max_height_index)

        param, covparam = curve_fit(self.linear_function, x_index_array, max_height_index_array)
        angle = np.degrees(np.arctan(param[0]))

        return angle, param, x_index_array, max_height_index_array

    def rotate_image(self, image, input_side="left"):
        # Function to rotate images using left/right/flip commands. The script is written with input on the left.
        if input_side == "left":
            image = image.rotate(90, expand=True)

        elif input_side == "right":
            image = image.rotate(-90, expand=True)

        elif input_side == "flip":
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        else:
            raise Exception("Specify input side 'left', 'right', 'flip'")

        return image

    def find_minimum_area_under_points(self, alpha_values, indents, w):
        # Finds the index with the minimum area under the points
        smoothed = savgol_filter(alpha_values, w, 1, mode="nearest")

        diff_indexes = np.where(np.abs(np.diff(smoothed)) < 1)[0]
        area_under_curve = smoothed[diff_indexes] * indents[diff_indexes]
        minimum_indent = diff_indexes[np.argmin(area_under_curve)]

        if self.show_plots:
            plt.figure(figsize=(10, 6))
            plt.plot(indents, alpha_values)
            plt.plot(indents, smoothed)
            plt.axvline(indents[minimum_indent], color='r', linestyle="dashed", label="minimum_index")
            plt.xlabel("Left indent")
            plt.ylabel("$\\alpha$ (dB/cm)")
            plt.legend(["$\\alpha$ values", "Smoothed alpha values", "Optimal indent: " + str(indents[minimum_indent])])
            plt.title(f"Left indent convergence with optimal indent: {indents[minimum_indent]}")
        return indents[minimum_indent]

    def find_optimal_left_indent(self, image, right_indent, waveguide_sum_width, IQR_neighbor_removal):
        left_indents = np.arange(100, 500, 4)
        converge_alpha_left_indent = []
        plot_state = self.show_plots
        if plot_state:
            self.show_plots = False
        for i in range(len(left_indents)):
            alpha_dB, r_squared, fit_x, fit_y, alpha_dB_variance = self.analyze_image(image, left_indents[i],
                                                                                      right_indent, waveguide_sum_width,
                                                                                      IQR_neighbor_removal)
            converge_alpha_left_indent.append(alpha_dB)
        self.show_plots = plot_state
        dI = left_indents[1] - left_indents[0]
        alpha_indent = np.gradient(converge_alpha_left_indent, dI)
        smoothed = savgol_filter(alpha_indent, 4, 1, mode="nearest")
        zero_gradient_minus = []
        zero_gradient_plus = []
        index_min = []
        abs_tol = 0.01
        while index_min and abs_tol < 0.101
                for i in range(len(left_indents) - 1):
                    zero_gradient_p = isclose(alpha_indent[i], alpha_indent[i + 1], abs_tol=tol)
                    zero_gradient_m = isclose(alpha_indent[i], alpha_indent[i - 1], abs_tol=tol)
                    zero_gradient_minus.append(zero_gradient_m)
                    zero_gradient_plus.append(zero_gradient_p)
                    if zero_gradient_plus[i] == True and zero_gradient_minus[i] == True:
                        index_min.append(left_indents[i])
                abs_tol = abs_tol + 0.01

        area_under_curve = smoothed[index_min] * left_indents[index_min]
        minimum_indent = index_min[np.argmin(area_under_curve)]
        if self.show_plots:
            plt.figure(figsize=(10, 6))
            # plt.plot(right_indents,converge_alpha_right_indent)
            plt.plot(left_indents, smoothed)
            plt.axvline(index_min[0], color='r', linestyle="dashed", label="minimum_index")
            plt.xlabel("Left indent")
            plt.ylabel("$d\\alpha$/d(indent)")
            plt.legend(["$\\alpha$ values", "Smoothed alpha values", "Optimal indent: " + str(index_min)])
            plt.title(f"Left indent convergence with optimal indent: " + str(index_min))
        return minimum_indent

    def find_optimal_right_indent(self, image, left_indent, waveguide_sum_width, IQR_neighbor_removal):
        right_indents = np.arange(100, 500, 4)
        converge_alpha_right_indent = []
        plot_state = self.show_plots
        if plot_state:
            self.show_plots = False
        for i in range(len(right_indents)):
            alpha_dB, r_squared, x_raw, y_raw, alpha_dB_variance = self.analyze_image(image, left_indent,right_indents[i],waveguide_sum_width,IQR_neighbor_removal)
            converge_alpha_right_indent.append(alpha_dB)
        self.show_plots = plot_state
        dI = right_indents[1] - right_indents[0]
        alpha_indent = np.gradient(converge_alpha_right_indent, dI)
        smoothed = savgol_filter(alpha_indent, 4, 1, mode="nearest")
        zero_gradient_minus = []
        zero_gradient_plus = []
        index_min = []
        abs_tol = 0.01
        for i in range(len(right_indents) - 1):
            zero_gradient_p = isclose(alpha_indent[i], alpha_indent[i + 1], abs_tol=abs_tol)
            zero_gradient_m = isclose(alpha_indent[i], alpha_indent[i - 1], abs_tol=abs_tol)
            zero_gradient_minus.append(zero_gradient_m)
            zero_gradient_plus.append(zero_gradient_p)
            if zero_gradient_plus[i] == True and zero_gradient_minus[i] == True:
                index_min.append(right_indents[i])
        if not index_min:
            abs_tol = abs_tol + 0.01
            i = 0
        if self.show_plots:
            plt.figure(figsize=(10, 6))
            #            plt.plot(right_indents,converge_alpha_right_indent)
            plt.plot(right_indents, smoothed)
            plt.axvline(index_min[0], color='r', linestyle="dashed", label="minimum_index")
            plt.xlabel("Right indent")
            plt.ylabel("$d\\alpha$/d(indent)")
            plt.legend(["$\\alpha$ values", "Smoothed alpha values", "Optimal indent: " + str(index_min)])
            plt.title(f"Right indent convergence with optimal indent: " + str(index_min))
        return index_min

    def find_optimal_waveguide_sum_width(self, image, left_indent, right_indent, IQR_neighbor_removal):
        rows = np.arange(30, 200, 10)
        converge_alpha_left_indent = []
        r_squared_list = []

        plot_state = self.show_plots
        if plot_state:
            self.show_plots = False

        for i in range(len(rows)):
            alpha_dB, r_squared, fit_x, fit_y, alpha_dB_variance = self.analyze_image(image, left_indent, right_indent,
                                                                                      rows[i], IQR_neighbor_removal)
            converge_alpha_left_indent.append(alpha_dB)
            r_squared_list.append(r_squared)

        self.show_plots = plot_state

        smoothed = savgol_filter(converge_alpha_left_indent, 5, 1, mode="nearest")
        optimal_width = rows[np.argmin(smoothed)]

        if self.show_plots:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            ax1.scatter(rows[np.argmin(smoothed)], np.min(smoothed), s=40, color="k",
                        label=f"Optimal wavegudie sum width: {rows[np.argmin(smoothed)]}")
            ax1.plot(rows, converge_alpha_left_indent, "r-", label="Alpha values")
            ax1.plot(rows, smoothed, "b-", label="Smoothed Alpha values")
            ax2.plot(rows, r_squared_list, "y-", label="Rsquared values")
            ax1.set_xlabel("waveguide sum width")
            ax1.set_ylabel("Alpha (dB/cm)")
            ax2.set_ylabel("R squared")
            ax1.legend(loc=2)
            ax2.legend(loc=1)

        return optimal_width

    def remove_outliers_IQR(self, x, data, blocks, num_neighbors):
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

    def linear_function(self, x, a, b):
        return a * x + b

    def exponential_function_offset(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def manual_input_and_output(self, input_point, output_point):
        # For spiral waveguides the input/output needs to be manually input.
        self.input_width_index = input_point[0]
        self.input_height_index = input_point[1]

        self.output_width_index = output_point[0]
        self.output_height_index = output_point[1]

    def calculate_confidence_interval(self, fit_parameters, fit_parameters_cov_var_matrix, x, confidence_interval):
        # Calculates the error of the fit parameters given a specified confidence interval
        var_a = fit_parameters_cov_var_matrix[0, 0]
        var_b = fit_parameters_cov_var_matrix[1, 1]
        var_c = fit_parameters_cov_var_matrix[2, 2]

        upper_a = fit_parameters[0] + confidence_interval * np.sqrt(var_a)
        lower_a = fit_parameters[0] - confidence_interval * np.sqrt(var_a)

        upper_b = fit_parameters[1] + confidence_interval * np.sqrt(var_b)
        lower_b = fit_parameters[1] - confidence_interval * np.sqrt(var_b)

        upper_c = fit_parameters[2] + confidence_interval * np.sqrt(var_c)
        lower_c = fit_parameters[2] - confidence_interval * np.sqrt(var_c)

        fit_upper = self.exponential_function_offset(x, *[upper_a, upper_b, upper_c])
        fit_lower = self.exponential_function_offset(x, *[lower_a, lower_b, lower_c])

        return fit_upper, fit_lower, upper_b, lower_b

    def crop_and_rotate(self, image, input_indent, output_indent, interval):
        # Crop and rotating of image to exclude input/output facet.
        image_array = np.asarray(image)

        if self.manual == True:
            input_width_index = self.input_width_index
            input_height_index = self.input_height_index
            output_width_index = self.output_width_index
            output_height_index = self.output_height_index
        else:
            input_width_index, input_height_index, output_width_index, output_height_index = self.insertion_detection(
                image.copy(), self.show_plots)

            input_point = (input_width_index, input_height_index)
            output_point = (output_width_index, output_height_index)
            self.set_um_per_pixel(input_point, output_point)

        window_num_pixel_height = np.shape(image_array)[1]  # 2048
        window_num_pixel_width = np.shape(image_array)[0]  # 2448

        # Cropping of image
        left_indent = input_width_index + input_indent
        right_indent = output_width_index - output_indent
        top_indent = input_height_index - (window_num_pixel_height / 20)

        if top_indent < 0:
            top_indent = 0

        bottom_indent = input_height_index + (window_num_pixel_height / 20)

        if bottom_indent > window_num_pixel_height:
            bottom_indent = window_num_pixel_height

        cropped_image = image.crop((left_indent, top_indent, right_indent, bottom_indent))
        cropped_image_array = np.asarray(cropped_image)

        # Find the waveguide and calculate angle of waveguide
        left_index_guess = 175

        number_of_points = 15

        separation = int((right_indent - left_indent - left_index_guess) / number_of_points)

        angle, angle_params, x_max_index_array, y_max_index_array = self.find_waveguide_angle(
            cropped_image_array[:, :, 2],
            left_index_guess, separation,
            number_of_points,
            self.show_plots)

        # Rotate picture and plot it with the upper and lower limit

        left_indent = left_indent
        right_indent = right_indent
        top_indent = top_indent
        bottom_indent = bottom_indent
        rotated_image = image.rotate(-angle, center=(left_indent, int(angle_params[1]) + top_indent)).crop(
            (left_indent, top_indent, right_indent, bottom_indent))

        rotated_image_array = np.asarray(rotated_image)

        upper = int(angle_params[1] + interval / 2)
        lower = int(angle_params[1] - interval / 2)

        # Convert x array unit from pixels to microns
        x_mu_array = np.arange(np.shape(rotated_image_array)[1]) * self.mum_per_pixel
        y_mu_array = np.arange(np.shape(rotated_image_array)[0]) * self.mum_per_pixel

        upper_index_array = (np.ones(len(rotated_image_array[0, :, 2])) * upper).astype("int")
        lower_index_array = (np.ones(len(rotated_image_array[0, :, 2])) * lower).astype("int")

        # if self.show_plots:
        # plt.figure(figsize=(10, 6))
        # plt.ylim([1000,2000])
        # plt.xticks([])
        # plt.yticks([])
        # plt.title("Original Image with cropped section")
        # plt.plot((left_indent, left_indent), (top_indent, bottom_indent), "r")
        # plt.plot((left_indent, right_indent), (bottom_indent, bottom_indent), "r")
        # plt.plot((left_indent, right_indent), (top_indent, top_indent), "r")
        # plt.plot((right_indent, right_indent), (top_indent, bottom_indent), "r")
        # plt.imshow(image)

        # plt.imshow(get_intensity_array(cropped_image_array.copy()), cmap="jet", vmin=0, vmax=10, interpolation='spline16', extent=[right_indent, left_indent, bottom_indent, top_indent])

        # plt.figure()
        # plt.imshow(self.get_intensity_array(cropped_image_array.copy()), cmap="jet", vmin=0, vmax=10,
        #           interpolation='spline16', extent=[x_mu_array[0], x_mu_array[-1], y_mu_array[0], y_mu_array[-1]])
        # plt.plot(x_mu_array[x_max_index_array], y_mu_array[y_max_index_array], 'r.')
        # plt.plot([x_mu_array[0], x_mu_array[-1]], [angle_params[1] * self.mum_per_pixel, (
        #            angle_params[0] * len(x_mu_array) + angle_params[1]) * self.mum_per_pixel], 'r-')
        # plt.title("Cropped")
        # plt.xlabel('x [um]')
        # plt.ylabel('y [um]')

        # Plot rotated picture
        # plt.figure()
        # plt.imshow(self.get_intensity_array(rotated_image_array), cmap="jet", vmin=0, vmax=10,
        #           extent=[x_mu_array[0], x_mu_array[-1], y_mu_array[0], y_mu_array[-1]])
        # plt.title("Rotated Image")
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlabel('x [um]')
        # plt.ylabel('y [um]')
        # plt.colorbar(fraction=0.016, pad=0.01)
        # plt.plot(x_mu_array[0:len(rotated_image_array[0, :, 2])], y_mu_array[upper_index_array], 'r-')
        # plt.plot(x_mu_array[0:len(rotated_image_array[0, :, 2])], y_mu_array[lower_index_array], 'r-')

        return rotated_image_array, x_mu_array, upper, lower

    def analyze_image(self, image, input_indent, output_indent, interval, num_neighbors):

        rotated_image_array, x_mu_array, upper, lower = self.crop_and_rotate(image, input_indent,
                                                                             output_indent, interval)

        cropped_image_height = np.shape(rotated_image_array)[0]

        x = x_mu_array

        # Sum channels to create intensity image
        image_data_raw = np.sum(rotated_image_array, 2)

        cropped_image = image_data_raw[cropped_image_height - upper: cropped_image_height - lower, :]

        # Sum values along the waveguide in an area
        y_raw = np.sum(cropped_image, axis=0)

        # Number of sections to split up the data for the IQR method
        smoothing = 10
        x_iqr, y_iqr, x_blocks = self.remove_outliers_IQR(x, y_raw, smoothing, num_neighbors)

        y_savgol = savgol_filter(y_iqr, 501, 1, mode="nearest")

        if self.show_plots:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y_raw, 'b-', label="Raw data")
            plt.legend()
            plt.xlabel('x Length [um]')
            plt.ylabel('Sum of pixel intensities')
            plt.show()

        fit_x = x_iqr
        fit_y = y_iqr

        initial_guess = [25, 0.0006, np.mean(fit_y[-10:])]
        bounds = ((0, 0, 0), (1000000, 1000000, 1000000))
        fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(
            self.exponential_function_offset,
            fit_x, fit_y, p0=initial_guess,
            full_output=True,
            maxfev=5000, bounds=bounds)  # sigma=weights, absolute_sigma=True
        # fit of exponential function with offset
        fit = self.exponential_function_offset(fit_x, fit_parameters[0], fit_parameters[1], fit_parameters[2])

        # fit_upper,fit_lower, alpha_upper, alpha_lower = self.calculate_confidence_interval(fit_parameters, fit_parameters_cov_var_matrix, fit_x, 1.960)

        residuals = fit_y - self.exponential_function_offset(fit_x, *fit_parameters)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((fit_y - np.mean(fit_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 1e4))
        alpha_dB_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4))
        # alpha_upper = 10 * np.log10(np.exp((alpha_upper) * 1e4))
        # alpha_lower = 10 * np.log10(np.exp((alpha_lower) * 1e4))

        if self.show_plots:
            font_size = 14

            plt.figure(figsize=(10, 6))
            plt.scatter(fit_x, fit_y, alpha=0.2, label="Outlier corrected data", s=4, color="k")
            plt.plot(fit_x, y_savgol, 'r-', label="Smoothed data")
            plt.plot(fit_x, fit, 'b-',
                     label=f"Fit to outlier corrected data: {alpha_dB:.1f} $\\pm$ {alpha_dB_variance:.1f} dB/cm")
            # plt.plot(fit_x, fit_upper, 'r', linestyle='dashed', label="95% Confidence Bound")
            # plt.plot(fit_x, fit_lower, 'r', linestyle='dashed')

            lgnd = plt.legend(fontsize=font_size, scatterpoints=1, frameon=False)
            lgnd.legendHandles[0]._sizes = [30]
            lgnd.legendHandles[0].set_alpha(1)
            plt.xlabel('x Length [um]')
            plt.ylabel('Sum of pixel intensity [a.u.]')
            plt.show()

            # plt.figure(figsize=(10, 6))
            # plt.scatter(x, y_raw, s=5 , color="b", label="Raw data")
            # plt.scatter(fit_x, fit_y, label="Outlier corrected data", s=5 , color="orange")
            # plt.plot(x_iqr, y_savgol, label="Savgol filter", color="r")
            # [plt.axvline(line,alpha=0.5) for line in x_blocks]
            # plt.xlabel('x Length [um]')
            # plt.ylabel('Sum of pixel intensity [a.u.]')
            # plt.xlim([0,x[-1]])
            # lgnd = plt.legend(fontsize=font_size,scatterpoints=1)
            # lgnd.legendHandles[0]._sizes = [30]
            # lgnd.legendHandles[0].set_alpha(1)
            # lgnd.legendHandles[1]._sizes = [30]
            # lgnd.legendHandles[1].set_alpha(1)
            # plt.show()

            # print("Fit Parameters:", fit_parameters)
            # print("Variance-Covariance Matrix Fit Parameters:", fit_parameters_cov_var_matrix)
            # print(
            #    f'a={fit_parameters[0]} +- {np.sqrt(fit_parameters_cov_var_matrix[0, 0])}, b={fit_parameters[1]} +- {np.sqrt(fit_parameters_cov_var_matrix[1, 1])}')
            # print(f"alpha = {fit_parameters[1] * 1e4} +- {np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4} 1/cm")
            # print(
            #    f'alpha_dB = {10 * np.log10(np.exp(fit_parameters[1] * 1e4))} +- {10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4))}) dB/cm')
            # print(f'Length over measurement = {fit_x[-1] - fit_x[0]} um')
            # print(f"R\u00b2 : {r_squared}")

        return alpha_dB, r_squared, fit_x, fit_y, alpha_dB_variance
