
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:08:25 2023

@authors: Magnus Linnet Madsen, Frederik Philip, Frederik Sørensen and Simon Sørensen
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, convolve2d
from math import isclose
import skimage.graph
from skimage.io import imread
from skimage.morphology import disk
from skimage.filters import rank
from skimage import util
import scipy.ndimage as ndi
import warnings
from skimage.color import rgb2gray
import matplotlib.patches as patches
#import cv2

warnings.filterwarnings('ignore')


class Camera:

    def __init__(self, device_number=0):
        self.__video_capture = cv2.VideoCapture(device_number)

    def __del__(self):
        self.__video_capture.release()

    def capture(self, filename=None):
        # Capture the video frame

        ret, frame = self.__video_capture.read()
        # frame = cv2.flip(frame,0)
        # frame = cv2.flip(frame,1)
        cv2.waitKey(1)
        if ret:
            if filename != None:
                cv2.imwrite(filename, frame)  # save frame as JPEG file
            return frame
        else:
            raise Exception("No Image frame acquired")

    def camsetup(self, width=2448, height=2048):
        # self.__video_capture.set(cv2.CAP_PROP_SETTINGS, 1)
        self.__video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.__video_capture.set(cv2.CAP_PROP_SETTINGS, 1)
        if not self.__video_capture.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = self.__video_capture.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # frame = cv2.flip(frame,0)
            # frame = cv2.flip(frame,1)
            resized_frame = cv2.resize(frame, (1600, 900))
            cv2.imshow('frame', resized_frame)

            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        cv2.destroyAllWindows()


class SPA:
    def __init__(self, show_plots, chiplength, manual=False):
        self.show_plots = show_plots
        self.chiplength = chiplength
        self.manual = manual  # Allows for manual input of input/output by calling and set_um_per_pixel before analyze_image

    def manual_input_and_output(self, input_point, output_point):
        # For spiral waveguides the input/output needs to be manually input.
        self.input_width_index = input_point[0]
        self.input_height_index = input_point[1]

        self.output_width_index = output_point[0]
        self.output_height_index = output_point[1]

    def set_um_per_pixel(self, point1, point2):
        # For straight waveguides: calculating Euclidean distance between input and output.
        points = [np.array(point1), np.array(point2)]
        dist_pixels = np.linalg.norm(points[0] - points[1])

        self.mum_per_pixel = self.chiplength / dist_pixels

    def get_intensity_array(self, image_array):
        # Convert array values to 8-bit compatible values.
        return np.clip(
            np.sqrt(image_array[:, :, 0] ** 2 + image_array[:, :, 1] ** 2 + image_array[:, :, 2] ** 2) / np.sqrt(
                3 * 255 ** 2) * 255, 0, 255)

    def insertion_detection(self, image):
        # Convert the image to a NumPy array
        image_array = np.array(image)
        image_array_shape = np.shape(image_array)

        scale_factor = 2 ** 3
        new_height = int(np.rint(image_array_shape[0] / scale_factor))  # New width in pixels
        new_width = int(np.rint(image_array_shape[1] / scale_factor))  # New height in pixels

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

        resized_input_index = np.unravel_index(np.argmax(intensity_filtered_image_array, axis=None),
                                               intensity_filtered_image_array.shape)

        input_height_index = int(resized_input_index[0] * scale_factor)
        input_width_index = int(resized_input_index[1] * scale_factor)

        height_tolerance_output = 100
        left_index_limit = int(new_width) - 20
        right_index_limit = int(new_width) - 1
        lower_index_limit = int((input_height_index - height_tolerance_output) / scale_factor)
        upper_index_limit = int((input_height_index + height_tolerance_output) / scale_factor)

        output_array = intensity_filtered_image_array[lower_index_limit: upper_index_limit,
                       left_index_limit: right_index_limit]
        resized_output_index = np.unravel_index(np.argmax(output_array, axis=None), output_array.shape)

        output_height_index = int((resized_output_index[0] + lower_index_limit) * scale_factor)
        output_width_index = int((resized_output_index[1] + left_index_limit) * scale_factor)

        return input_width_index, input_height_index, output_width_index, output_height_index

    def find_waveguide_angle(self, image_array, left_index_guess, left_right_separation, number_of_points):
        # Find the angle of the waveguide and rotate the image to ensure it is always horizontal.
        # Define kernel for convolution
        kernel = np.ones([1, 50]) / (1 * 50)
        smoothed_image_array = convolve2d(image_array, kernel)

        # Define the position of the waveguide
        x_index_array = []
        max_height_index_array = []
        for index in range(0, number_of_points):
            x_index = left_index_guess + index * left_right_separation
            x_index_array.append(x_index)
            max_array = np.flip(np.mean(smoothed_image_array[:, x_index: left_index_guess + x_index + 1], axis=1))
            max_height_index = np.argmax(max_array - np.mean(max_array))
            max_height_index_array.append(max_height_index)
        # Fit a linear function between input/output and find the angle
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

    def optimize_parameter(self, parameter, image, left_crop, right_crop, waveguide_sum_width,
                           IQR_neighbor_removal,invalid_index):
        # Optimizing the parameters used in the straight waveguide fit
        plot_state = self.show_plots
        if plot_state:
            self.show_plots = False
        converge_alpha = []

        # Fitting for varying left/right crop and sum widths
        if parameter == "left crop":
            crop = np.arange(10, 350, 2)
            for i in range(len(crop)):
                alpha_dB, r_squared, alpha_dB_variance = self.analyze_image(image, crop[i], right_crop,
                                                                                waveguide_sum_width, IQR_neighbor_removal)
                converge_alpha.append(alpha_dB)

        elif parameter == "right crop":
            crop = np.arange(10, 350, 2)
            for i in range(len(crop)):
                alpha_dB, r_squared, alpha_dB_variance = self.analyze_image(image, left_crop, crop[i],
                                                                            waveguide_sum_width, IQR_neighbor_removal)
                converge_alpha.append(alpha_dB)

        elif parameter == "sum width":
            crop = np.arange(10, 150, 2)
            for i in range(len(crop)):
                alpha_dB, r_squared, alpha_dB_variance = self.analyze_image(image, left_crop, right_crop,
                                                                            crop[i], IQR_neighbor_removal)
                converge_alpha.append(alpha_dB)

        else:
            raise Exception("Specify parameter 'left crop', 'right crop' or 'sum width'")

        # differentiate and smooth the determined loss values
        dI = crop[1] - crop[0]
        smoothed_alpha = savgol_filter(converge_alpha, 4, 1, mode="nearest")
        alpha_crop = np.gradient(smoothed_alpha, dI)

        # Remove divergent points in the differentiated data
        mask = np.abs(alpha_crop) <= 2.5

        alpha_crop = alpha_crop[mask]
        crop = crop[mask]

        # Findng points below threshold
        index_min = []
        abs_tol = 0.01
        while abs_tol < 0.9:
            if len(index_min) == 0:
                zero_gradient_minus = []
                zero_gradient_plus = []
                for i in range(len(crop) - 1):
                    zero_gradient_p = isclose(alpha_crop[i], alpha_crop[i + 1], abs_tol=abs_tol)
                    zero_gradient_m = isclose(alpha_crop[i], alpha_crop[i - 1], abs_tol=abs_tol)
                    zero_gradient_minus.append(zero_gradient_m)
                    zero_gradient_plus.append(zero_gradient_p)
                    if zero_gradient_plus[i] == True and zero_gradient_minus[i] == True:
                        index_min.append(i)
            abs_tol = abs_tol + 0.01

        # Finding points where the variation in the surrounding points are ~0.
        num_neighbors = 5
        point_mean = []
        for index in index_min:
            neighbor_indexes = np.arange(index - num_neighbors, index + num_neighbors + 1, 1)
            neighbor_indexes = [x for x in neighbor_indexes if x > 0 and x < len(alpha_crop)]
            point_m = np.mean(smoothed_alpha[neighbor_indexes])
            point_diff = abs(point_m - smoothed_alpha[index])
            point_mean.append(point_diff)

        # Finding the point where the variation in adjacent points is minimum
        if invalid_index:
            del point_mean[invalid_index]

        min_point_mean = point_mean.index(min(point_mean))
        ideal_crop = crop[index_min[min_point_mean]]

        self.show_plots = plot_state
        if self.show_plots:
            plt.figure(figsize=(10, 6))
            plt.plot(crop, alpha_crop)
            plt.axvline(ideal_crop, color='r', linestyle="dashed", label="minimum_index")
            plt.xlabel(parameter, fontsize=20)
            plt.ylabel("$d\\alpha$/d(crop)", fontsize=20)
            plt.legend(["Smoothed $\\alpha$ values", "Optimal " + parameter + " " + str(ideal_crop)], fontsize=16)
            plt.show()

        return min_point_mean, ideal_crop

    def remove_outliers_IQR(self, x, data, subsets, num_neighbors):
        # Removal of outliers using the interquartile method.
        data_subsets = np.array_split(data, subsets)
        x_subsets = np.array_split(x, subsets)
        x_subsets_indexes = [x[-1] for x in x_subsets]

        for i in range(len(data_subsets)):
            Q1 = np.percentile(data_subsets[i], 25, interpolation='midpoint')
            Q3 = np.percentile(data_subsets[i], 75, interpolation='midpoint')
            IQR = Q3 - Q1

            upper = Q3 + 1.5 * IQR
            lower = Q1 - 1.5 * IQR

            upper_array = np.where(data_subsets[i] >= upper)[0]
            lower_array = np.where(data_subsets[i] <= lower)[0]

            remove_array = np.concatenate((upper_array, lower_array))
            new_remove_array = []

            for index in remove_array:  # Finding indexes of neighbors to detected outliers
                neighbor_indexes = np.arange(index - num_neighbors, index + num_neighbors + 1, 1)
                neighbor_indexes = [x for x in neighbor_indexes if x > 0 and x < len(data_subsets[i])]
                new_remove_array += neighbor_indexes

            new_remove_array = list(set(new_remove_array))
            data_subsets[i] = np.delete(data_subsets[i], new_remove_array)  # removing outliers and neighbors from data
            x_subsets[i] = np.delete(x_subsets[i], new_remove_array)

        return np.concatenate(x_subsets), np.concatenate(data_subsets), x_subsets_indexes

    def linear_function(self, x, a, b):
        return a * x + b

    def exponential_function_offset(self, x, a, b, c):
        return a * np.exp(-b * x) + c

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

    def crop_and_rotate(self, image, input_crop, output_crop, interval):
        # Crop and rotating of image to exclude input/output facet.
        image_array = np.asarray(image)

        if self.manual == True:
            input_width_index = self.input_width_index
            input_height_index = self.input_height_index
            output_width_index = self.output_width_index
        else:
            input_width_index, input_height_index, output_width_index, output_height_index = self.insertion_detection(
                image.copy())

            input_point = (input_width_index, input_height_index)
            output_point = (output_width_index, output_height_index)
            self.set_um_per_pixel(input_point, output_point)

        window_num_pixel_height = np.shape(image_array)[1]  # 2048

        # Cropping of image
        left_crop = input_width_index + input_crop
        right_crop = output_width_index - output_crop
        top_crop = input_height_index - (window_num_pixel_height / 20)

        if top_crop < 0:
            top_crop = 0

        bottom_crop = input_height_index + (window_num_pixel_height / 20)

        if bottom_crop > window_num_pixel_height:
            bottom_crop = window_num_pixel_height

        cropped_image = image.crop((left_crop, top_crop, right_crop, bottom_crop))
        cropped_image_array = np.asarray(cropped_image)

        # Find the waveguide and calculate angle of waveguide
        left_index_guess = 175

        number_of_points = 15

        separation = int((right_crop - left_crop - left_index_guess) / number_of_points)

        angle, angle_params, x_max_index_array, y_max_index_array = self.find_waveguide_angle(
            cropped_image_array[:, :, 2], left_index_guess, separation, number_of_points)

        # Rotate image
        left_crop = left_crop
        right_crop = right_crop
        top_crop = top_crop
        bottom_crop = bottom_crop
        rotated_image = image.rotate(-angle, center=(left_crop, int(angle_params[1]) + top_crop)).crop(
            (left_crop, top_crop, right_crop, bottom_crop))

        rotated_image_array = np.asarray(rotated_image)

        upper = int(angle_params[1] + interval / 2)
        lower = int(angle_params[1] - interval / 2)

        # Convert x array unit from pixels to microns
        x_mu_array = np.arange(np.shape(rotated_image_array)[1]) * self.mum_per_pixel

        return rotated_image_array, x_mu_array, upper, lower

    def analyze_image(self, image, input_crop, output_crop, interval, num_neighbors):
        # The fitting of the image
        rotated_image_array, x_mu_array, upper, lower = self.crop_and_rotate(image, input_crop, output_crop,
                                                                             interval)

        cropped_image_height = np.shape(rotated_image_array)[0]

        x = x_mu_array

        # Sum channels to create intensity image
        image_data_raw = np.sum(rotated_image_array, 2)

        cropped_image = image_data_raw[cropped_image_height - upper: cropped_image_height - lower, :]

        # Sum values along the waveguide in an area
        y_raw = np.sum(cropped_image, axis=0)

        # Removing outliers using the interquartile method
        num_subsets = 10
        x_iqr, y_iqr, x_subsets = self.remove_outliers_IQR(x, y_raw, num_subsets, num_neighbors)

        y_savgol = savgol_filter(y_iqr, 501, 1, mode="nearest")

        fit_x = x_iqr
        fit_y = y_iqr

        initial_guess = [25, 0.0006, np.mean(fit_y[-10:])]
        bounds = ((0, 0, 0), (1000000, 1000000, 1000000))
        fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(
            self.exponential_function_offset, fit_x, fit_y, p0=initial_guess, full_output=True, maxfev=5000,
            bounds=bounds)

        # fit of exponential function with offset
        fit = self.exponential_function_offset(fit_x, fit_parameters[0], fit_parameters[1], fit_parameters[2])

        residuals = fit_y - self.exponential_function_offset(fit_x, *fit_parameters)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((fit_y - np.mean(fit_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 1e4))
        alpha_dB_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4))
#
        initial_guess = [25, 0.0006, np.mean(y_raw[-10:])]
        bounds = ((0, 0, 0), (1000000, 1000000, 1000000))
        fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(
            self.exponential_function_offset, x, y_raw, p0=initial_guess, full_output=True, maxfev=5000,
            bounds=bounds)

        # fit of exponential function with offset
        fit_raw = self.exponential_function_offset(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])

        residuals = y_raw - self.exponential_function_offset(x, *fit_parameters)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
        r_squared_raw = 1 - (ss_res / ss_tot)

        alpha_dB_raw = 10 * np.log10(np.exp(fit_parameters[1] * 1e4))
        alpha_dB_variance_raw = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 1e4))

        x_iqr = x_iqr / 1000
        x = x/1000

        if self.show_plots:
            font_size = 24
            plt.figure(figsize=(9.5, 6.5))
            plt.plot(x_iqr, fit, color="#E69F00", linestyle="-", linewidth=3,
                     label=f"{alpha_dB:.1f}$\\pm${alpha_dB_variance:.1f} dB/cm, R\u00b2: {r_squared:.2f}")  # ,
            plt.plot(x, fit_raw, color="g", linestyle="-", linewidth=3,
                     label=f"{alpha_dB_raw:.1f}$\\pm${alpha_dB_variance_raw:.1f} dB/cm, R\u00b2: {r_squared_raw:.2f}")  # ,
            plt.scatter(x, y_raw, color="#0072B2", s=3)#, label="Raw data")
            plt.scatter(x_iqr, y_iqr, color="#000000", s=3)#, label="Outlier corrected data")

            lgnd = plt.legend(fontsize=font_size, scatterpoints=1, frameon=False)
#            lgnd.legendHandles[2]._sizes = [30]
#            lgnd.legendHandles[2].set_alpha(1)
#            lgnd.legendHandles[3]._sizes = [30]
#            lgnd.legendHandles[3].set_alpha(1)
            plt.xlabel('Propagation length [mm]', fontsize=font_size)
            plt.ylabel('Mean intensity [a.u.]', fontsize=font_size)
            plt.xlim([min(x_iqr), max(x)])
            plt.ylim([min(y_raw), max(y_raw) + 5])
            plt.xticks([1,2,3],fontsize=font_size)
            plt.yticks([1000,3000,5000,7000],fontsize=font_size)
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        plt.show()

        return alpha_dB, r_squared, alpha_dB_variance

    def three_dimension_plot(self,img,manual,x_offset,y_offset):
        #Convert the image to np.array and then to grayscale for intensity to be normalized.
        image_array = np.array(img)

        image = rgb2gray(image_array)
        #Input / ouput can be manually input or automatically detected.
        if manual == True:
            x_start = np.int32(input("Enter first x: "))
            y_start = np.int32(input("Enter first y: "))
            x_end = np.int32(input("Enter last x: "))
            y_end = np.int32(input("Enter last y: "))
        else:
            x_start, y_start, x_end, y_end = self.insertion_detection(img.copy())

        #The input and output are printed
        in_point = [x_start,y_start]
        out_point = [x_end,y_end]
        print("Input coordinates: ", in_point)
        print("Output coordinates: ", out_point)

        # Define the cropped image and allows for x and y offset as the input will include facet scattering.
        x_values = np.arange(x_start + x_offset, x_end - x_offset)
        y_values = np.arange(min(y_start, y_end) - y_offset, max(y_start, y_end) - y_offset)

        # Crop the image
        cropped_image = image[y_values[0]:y_values[-1], x_values[0]:x_values[-1]]

        #Plot the full image with a rectangle patch indicating the cropped region
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        #Show input and output
        plt.scatter(*in_point, color='red', marker='o', label='In Point')
        plt.scatter(*out_point, color='blue', marker='o', label='Out Point')
        # Create a rectangle patch
        rect = patches.Rectangle((x_values[0], y_values[0]), x_values[-1] - x_values[0], y_values[-1] - y_values[0],
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)
        plt.legend()
        #plt.show()

        #Construct the 3D plot
        z = np.asarray(cropped_image)*255
        mydata = z[::1, ::1]
        font_size = 16
        label_pad = 10
        fig = plt.figure(facecolor='w')
        ax2 = fig.add_subplot(1, 1, 1, projection='3d')
        x, y = np.mgrid[:mydata.shape[0], :mydata.shape[1]]
        ax2.plot_surface(x, y, mydata, cmap=plt.cm.turbo, rstride=1, cstride=1, linewidth=0., antialiased=False)
        ax2.set_zlim3d(0, 255)
        #Rotate the plot
        ax2.view_init(elev=30, azim=25)#210
        ax2.set_xlabel('Width [pixels]', fontsize=font_size, labelpad = label_pad)
        ax2.set_ylabel('Length [pixels]', fontsize=font_size, labelpad = label_pad)
        ax2.set_zlabel('Norm. intensity', fontsize=font_size, labelpad = label_pad)

        ax2.set_yticks([0, 500, 1000])
        ax2.set_zticks([0,128,255])
        ax2.tick_params(axis='both', which='major', labelsize=font_size)  # Adjust labelsize as needed
        ax2.tick_params(axis='z', which='major', labelsize=font_size, pad = 5)  # Adjust z-axis labelsize as needed

        ax2.set_box_aspect([1, 2, 1.5])  # Aspect ratio is width:length:height

        plt.show()

    def straight_waveguide(self, image, optimize_parameter):
        IQR_neighbor_removal = 1
        if optimize_parameter:
            input_crop = self.optimize_parameter("left crop", image, 200, 100, 80, IQR_neighbor_removal)
            output_crop = self.optimize_parameter("right crop", image, 200, 100, 80, IQR_neighbor_removal)
            interval = self.optimize_parameter("sum width", image, 200, 100, 80, IQR_neighbor_removal)
        else:
            input_crop = np.int32(input("Enter left crop: "))
            output_crop = np.int32(input("Enter right crop: "))
            interval = np.int32(input("Enter sum width: "))
            IQR_neighbor_removal = np.int32(input("Enter width of removal of IQR: "))

        return self.analyze_image(image, input_crop, output_crop, interval, IQR_neighbor_removal)

    ################################### SPIRAL #######################################

    def mean_image_intensity(self, image, mum_per_pixel, in_point, out_point):
        # Meaning the image
        disk_size = 20
        mean_disk = disk(disk_size)

        mean_image = (rank.mean_percentile(image, footprint=mean_disk, p0=0, p1=1))
        x_path, y_path = self.path_finder(0.1, in_point, out_point, image, mum_per_pixel)
        y_raw = mean_image[y_path, x_path]

        x_image = range(len(y_raw))

        x = np.array([x * mum_per_pixel for x in x_image])

        return x, y_raw


    def find_path(self, bw_image, start, end):
        # Converting the black-white image to a cost path matrix
        costs = np.where(bw_image == 1, 1, 10000)
        path, cost = skimage.graph.route_through_array(costs, start=start, end=end, fully_connected=True,
                                                       geometric=True)
        return path, cost

    def initialize(self):
        # Define class attributes for click-related variables
        self.first_click = None
        self.second_click = None
        self.click_count = 0
        self.window_opened = False  # Flag to track if the window is opened

    def reset(self):
        # Reset click-related variables to their initial states
        self.first_click = None
        self.second_click = None
        self.click_count = 0

    def get_click_coordinates(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_count == 0:
                self.first_click = (x, y)
                self.click_count += 1
            elif self.click_count == 1:
                self.second_click = (x, y)
                self.click_count += 1

            if self.click_count >= 2:
                return (self.first_click, self.second_click)  # Return coordinates after both clicks

    def run(self, image_path, scale_factor):
        # Initialize class attributes
        self.initialize()

        # Load the image
        image = cv2.imread(image_path)

        # Resize the image based on the scale factor
        if scale_factor != 1.0:
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (width, height))

        # Display the image if the window is not already opened
        if not self.window_opened:
            cv2.imshow('Image', image)
            self.window_opened = True

        # Function to handle mouse events and store coordinates
        def handle_mouse_event(event, x, y, flags, param):
            self.get_click_coordinates(event, x, y)  # Call the actual method

        # Set mouse callback function to handle events
        cv2.setMouseCallback('Image', handle_mouse_event)

        # Wait indefinitely for two mouse click events
        while self.click_count < 2:
            cv2.waitKey(1)

        # Close the window after the second click event
        cv2.destroyAllWindows()

        # Process the click coordinates after both clicks
        x1, y1 = self.first_click
        x2, y2 = self.second_click

        # Scale the coordinates based on the scale factor
        x1_scaled = int(x1 / scale_factor)
        y1_scaled = int(y1 / scale_factor)
        x2_scaled = int(x2 / scale_factor)
        y2_scaled = int(y2 / scale_factor)

        # Construct input and output points
        input_point = [x1_scaled, y1_scaled]
        output_point = [x2_scaled, y2_scaled]

        # Return the scaled coordinates of the first and second clicks
        return input_point, output_point


    def grey_image(self, path):
        # Determining the input/output facet
        image = util.img_as_float(imread(path))
        grey_image = image[:, :, 2]

        return grey_image

    def um_per_pixel(self, point1, point2, distance):
        # calculating Euclidean distance
        dist_pixels = np.linalg.norm(point1 - point2)
        return distance / dist_pixels

    def opt_crop(self, parameter, x_iqr, y_iqr):
        # Optimizing the parameters used in the fitting of the spiral waveguides
        font_size = 18
        y_savgol = savgol_filter(y_iqr, 2000, 1)
        # Determining the alpha values for fits of varying parameters.
        crop = np.arange(0, 800, 20)
        dI = crop[1] - crop[0]
        alpha_dB_i = []
        if parameter == "left crop":
            for i in range(1, len(crop)):
                x_iqr = x_iqr[i:]
                y_savgol = y_savgol[i:]
                fit_x = x_iqr
                intensity_values = y_savgol
                initial_guess = [25, 0.0006, np.min(intensity_values)]
                fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(
                    self.exponential_function_offset, fit_x, intensity_values, p0=initial_guess, full_output=True,
                    maxfev=5000)
                alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 10))
                alpha_dB_i.append(alpha_dB)

        elif parameter == "right crop":
            for i in range(1, len(crop)):  # 525
                x_iqr = x_iqr[:-i]
                y_savgol = y_savgol[:-i]
                fit_x = x_iqr
                intensity_values = y_savgol
                initial_guess = [25, 0.0006, np.min(intensity_values)]
                fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(
                    self.exponential_function_offset, fit_x, intensity_values, p0=initial_guess, full_output=True,
                    maxfev=5000)
                alpha_dB = 10 * np.log10(np.exp(fit_parameters[1] * 10))
                alpha_dB_i.append(alpha_dB)
        # For the smoothed alpha values find points where the gradient is ~0
        smoothed_alpha = savgol_filter(alpha_dB_i, 4, 1, mode='nearest')
        alpha_crop = np.gradient(smoothed_alpha, dI)
        index_min = []
        abs_tol = 0.01
        while abs_tol < 0.9:
            if len(index_min) == 0:
                zero_gradient_minus = []
                zero_gradient_plus = []
                for i in range(len(alpha_crop) - 1):
                    zero_gradient_p = isclose(alpha_crop[i], alpha_crop[i + 1], abs_tol=abs_tol)
                    zero_gradient_m = isclose(alpha_crop[i], alpha_crop[i - 1], abs_tol=abs_tol)
                    zero_gradient_minus.append(zero_gradient_m)
                    zero_gradient_plus.append(zero_gradient_p)
                    if zero_gradient_plus[i] == True and zero_gradient_minus[i] == True:
                        index_min.append(i)
            abs_tol = abs_tol + 0.01
        # Comparing the previously found points to surrounding points and findning the minimum
        num_neighbors = 5
        point_mean = []
        for index in index_min:
            neighbor_indexes = np.arange(index - num_neighbors, index + num_neighbors + 1, 1)
            neighbor_indexes = [x for x in neighbor_indexes if x > 0 and x < len(alpha_crop)]
            point_m = np.mean(smoothed_alpha[neighbor_indexes])
            point_diff = abs(point_m - smoothed_alpha[index])
            point_mean.append(point_diff)
        min_point_mean = point_mean.index(min(point_mean))
        ideal_crop = crop[index_min[min_point_mean]]
        if self.show_plots:
            plt.figure(figsize=(10, 6))
            plt.plot(crop[:-1], alpha_crop, "k")
            plt.xlabel(parameter, fontsize=font_size)
            plt.ylabel("$d\\alpha$/d(crop)", fontsize=font_size)
            plt.axvline(ideal_crop, color="r", linestyle="--",
                        label="Optimized " + parameter + " " + str(ideal_crop))
            plt.legend(["Smoothed $\\alpha$ values", "Optimal crop: " + str(ideal_crop)], fontsize=font_size)
            plt.show()
        return ideal_crop

    def path_finder(self, threshold, in_point, out_point, grey_image, mum_per_pixel):
        # Using the cost path image to find the optimal path
        sobel_h = ndi.sobel(grey_image, 0)
        sobel_v = ndi.sobel(grey_image, 1)
        magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
        font_size = 18
        path_length = []
        threshold = np.round(np.linspace(0.01, threshold, 10), 2)
        # Using different threshold values to find different path lengths
        for i in threshold:
            bw_waveguide = grey_image > i
            start = (in_point[1], in_point[0])
            end = (out_point[1], out_point[0])
            path, costs = self.find_path(bw_waveguide, start, end)
            path_length.append(path)
        # Finding the longest path length as it is the correct path
        diff_paths = []
        path_length_mum = []
        for element in path_length:
            sub_length = len(element)
            length_mum = sub_length * mum_per_pixel
            diff_paths.append(sub_length)
            path_length_mum.append(length_mum)

        max_element = max(path_length_mum)
        max_index = path_length_mum.index(max_element)

        x_path = []
        y_path = []
        for i in range(len(path_length[max_index])):
            x_path.append(path_length[max_index][i][1])
            y_path.append(path_length[max_index][i][0])

        plt.figure(figsize=(8,6))
        if self.show_plots:
            plt.plot(*in_point, "ro")
            plt.plot(*out_point, "ro")
            plt.scatter(x_path[::100], y_path[::100], s=32, alpha=1, color="red")
            plt.xlabel("Width [a.u.]", fontsize=font_size)
            plt.ylabel("Height [a.u.]", fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            # plt.axis('off')
            plt.imshow(magnitude, cmap="turbo")
        return x_path, y_path

    def spiral_fit(self, x_iqr, y_iqr, x, y_raw, l, r):
        font_size = 18
        x_iqr = x_iqr[l:-r]
        y_iqr = y_iqr[l:-r]
        x = x[l:-r]
        y_raw = y_raw[l:-r]

        fit_x = x_iqr

        # Fit to outlier corrected data
        initial_guess = [25, 0.0006, np.min(y_iqr)]
        fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(self.exponential_function_offset,
                                                                                        x_iqr,
                                                                                        y_iqr, p0=initial_guess,
                                                                                        full_output=True,
                                                                                        maxfev=5000)  # sigma=weights, absolute_sigma=True
        fit_outlier = self.exponential_function_offset(x_iqr, fit_parameters[0], fit_parameters[1], fit_parameters[2])

        residuals = y_iqr - self.exponential_function_offset(fit_x, *fit_parameters)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_iqr - np.mean(y_iqr)) ** 2)
        r_squared_outlier = 1 - (ss_res / ss_tot)

        alpha_dB_outlier = 10 * np.log10(np.exp(fit_parameters[1] * 10))
        alpha_dB_outlier_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 10))

        # Fit to raw data
        initial_guess = [25, 0.0006, np.min(y_raw)]
        fit_parameters, fit_parameters_cov_var_matrix, infodict, mesg, ier, = curve_fit(self.exponential_function_offset, x,
                                                                                        y_raw, p0=initial_guess,
                                                                                        full_output=True,
                                                                                        maxfev=5000)  # sigma=weights, absolute_sigma=True
        fit_raw = self.exponential_function_offset(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
        residuals = y_raw - self.exponential_function_offset(x, *fit_parameters)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
        r_squared_raw = 1 - (ss_res / ss_tot)

        alpha_dB_raw = 10 * np.log10(np.exp(fit_parameters[1] * 10))
        alpha_dB_raw_variance = 10 * np.log10(np.exp(np.sqrt(fit_parameters_cov_var_matrix[1, 1]) * 10))
        if self.show_plots:
            plt.figure(figsize=(9,6))
            plt.plot(x_iqr, fit_outlier, color="#E69F00", linestyle="-", linewidth=3,
                     label=f"{alpha_dB_outlier:.1f}$\\pm${alpha_dB_outlier_variance:.1f} dB/cm, R\u00b2: {r_squared_outlier:.2f}")  # ,
            plt.plot(x, fit_raw, color="g", linestyle="-", linewidth=3,
                     label=f"{alpha_dB_raw:.1f}$\\pm${alpha_dB_raw_variance:.1f} dB/cm, R\u00b2: {r_squared_raw:.2f}")  # ,
            plt.scatter(x, y_raw, color="#0072B2", s=3, label="Raw data")
            plt.scatter(x_iqr, y_iqr, color="#000000", s=3, label="Outlier corrected data")
            lgnd = plt.legend(fontsize=font_size, scatterpoints=1, frameon=False)
            lgnd.legendHandles[2]._sizes = [30]
            lgnd.legendHandles[2].set_alpha(1)
            lgnd.legendHandles[3]._sizes = [30]
            lgnd.legendHandles[3].set_alpha(1)
            plt.xlabel('Propagation length [mm]', fontsize=font_size)
            plt.ylabel('Mean intensity [a.u.]', fontsize=font_size)
            plt.xlim([min(x_iqr), max(x)])
            plt.ylim([min(y_raw), max(y_raw) + 5])
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
        plt.show()

        return alpha_dB_outlier, alpha_dB_outlier_variance, r_squared_outlier, alpha_dB_raw, alpha_dB_raw_variance, r_squared_raw

    def spiral_waveguide(self, image_directory, distance_um, parameter_optimize,scale_factor):
        path = image_directory
        grey_image = self.grey_image(path)

        in_point, out_point = self.run(image_directory, scale_factor=scale_factor)
        print("Input coordinates: ", in_point)
        print("Output coordinates: ", out_point)
        point1 = np.array((in_point[0], in_point[1]))
        point2 = np.array((in_point[0], out_point[1]))  # 1985

        mum_per_pixel = self.um_per_pixel(point1, point2, distance_um)

        x, y_raw = self.mean_image_intensity(grey_image, mum_per_pixel, in_point, out_point)

        x_iqr, y_iqr, indexes = self.remove_outliers_IQR(x, y_raw, 10, 1)
        if parameter_optimize:
            l = self.opt_crop("left crop", x_iqr, y_iqr)
            r = self.opt_crop("right crop", x_iqr, y_iqr)
        else:
            l = np.int32(input("Enter left crop: "))
            r = np.int32(input("Enter right crop: "))

        alpha_dB_outlier, alpha_dB_outlier_variance, r_squared_outlier, alpha_dB_raw, alpha_dB_raw_variance, r_squared_raw = self.spiral_fit(
            x_iqr, y_iqr, x, y_raw, l, r)

        return alpha_dB_outlier, alpha_dB_outlier_variance, r_squared_outlier, alpha_dB_raw, alpha_dB_raw_variance, r_squared_raw
