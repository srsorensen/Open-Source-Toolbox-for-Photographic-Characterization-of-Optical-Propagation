# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:20:07 2023

@author: frede
"""
import traceback
from PIL import Image
import sys
sys.path.append("C:/Users/shd-PhotonicLab/PycharmProjects/SPA/") #Path to repository with SPA.py and Lab_cam.py
from SPA import SPA
import os
import h5py
import numpy as np

def get_wavelength(filename):
    return str(np.round(float(filename.split(".bmp")[0].split("_")[-1][:-2]),1))


path = 'E:/AlGaAs_Power/'
f_ending = '.bmp'
contains = 'ST3'



image_path_dict = {}


for image_dir in os.listdir(path):
    if os.path.isdir(path + image_dir) and contains in image_dir:
        filenames = []
        for file in os.listdir(path + image_dir):
            if file.endswith(f_ending):
                filenames.append(file)
        image_path_dict[image_dir] = filenames

print(image_path_dict.keys())

test_image = list(image_path_dict.values())[0]
test_path = list(image_path_dict.keys())[0]
image = Image.open(path + test_path + "/" + image_path_dict[test_path][0])

chip_length = 4870
spa = SPA(False,chip_length) #set flag to False to turn off plotting,

left_indent = 200
right_indent = 100
waveguide_sum_width = 80
IQR_neighbor_removal = 1
sum_width = 80

image = spa.rotate_image(image, "flip")
print(spa.analyze_image(image,left_indent,right_indent,waveguide_sum_width,IQR_neighbor_removal))

choice = input("q for quit enter to continue: ")

if choice == "q":
    print("Exiting code")
else:
    spa.show_plots = False
    counter = 0
    for key in image_path_dict.keys():
        print(key)
        files = image_path_dict[key]
        wavelengths = []
        r_squared_values = []
        alphas = []
        alpha_variances = []
        right_indent_sweep = []
        left_indent_sweep = []
        sum_width_sweep = []

        # Initialize lists to keep track of invalid indices
        invalid_sum_indices = []
        invalid_left_indices = []
        invalid_right_indices = []

        for file in files:
            valid_alpha_found = False  # Flag to track if a valid alpha has been found
            retries = 0  # To track the number of retries

            while retries < 3:  # Attempt up to 3 times
                try:
                    image = Image.open(path + key + "/" + file)
                    image = spa.rotate_image(image, "flip")
                    if retries == 0:
                        # Calculate all indices on the first attempt or when adjusting the sum index
                        sum_index, point_mean_sum, sum_width_opt = spa.optimize_parameter(
                            "sum width", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            invalid_sum_indices
                        )
                        left_index, point_mean_left, left_indent_opt = spa.optimize_parameter(
                            "left crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            invalid_left_indices
                        )
                        right_index, point_mean_right, right_indent_opt = spa.optimize_parameter(
                            "right crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            invalid_right_indices
                        )
                    elif retries == 1:
                        # Recalculate only the left index and related values
                        left_index, point_mean_left, left_indent_opt = spa.optimize_parameter(
                            "left crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            invalid_left_indices
                        )
                        print(left_indent_opt)
                        print(left_index)

                    elif retries == 2:
                        # Recalculate only the right index and related values
                        right_index, point_mean_right, right_indent_opt = spa.optimize_parameter(
                            "right crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            invalid_right_indices
                        )

                    # Analyze the image to get the alpha value
                    alpha, rsquared, alpha_variance = spa.analyze_image(
                        image, left_indent_opt, right_indent_opt, sum_width_opt, IQR_neighbor_removal
                    )
                    print(alpha)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    traceback.print_exc()
                    break  # Skip to next file if an exception occurs
                finally:
                    image.close()

                # Check if alpha is valid
                if alpha <= 0 or alpha > 200 or np.isinf(alpha) or np.isnan(alpha):
                    # Check point_mean and adjust indices if necessary
                    if retries == 0:
                        if point_mean_sum > 1:
                            if sum_index not in invalid_sum_indices:
                                invalid_sum_indices.append(sum_index)
                        else:
                            retries =+ 1
                    elif retries == 1:
                        if point_mean_left > 1:
                            if left_index not in invalid_left_indices:
                                invalid_left_indices.append(left_index)
                        else:
                            retries =+ 1
                    elif retries == 2:
                        if point_mean_right > 1:
                            if right_index not in invalid_right_indices:
                                invalid_right_indices.append(right_index)
                        else:
                            retries =+ 1

                        # Retry the current index adjustment
                    else:
                        print('All options exhausted for this file.')
                        break  # Exit the while loop if no valid options are left

                else:
                    # If alpha is valid, store the results and move to the next file
                    print(
                        f"{counter}, left indent: {left_indent_opt}, right indent: {right_indent_opt}, sum width: {sum_width_opt}, Loss: {np.round(alpha, 1)} dB/cm")
                    counter += 1
                    wavelength = get_wavelength(file)
                    alphas.append(alpha)
                    alpha_variances.append(alpha_variance)
                    wavelengths.append(wavelength)
                    r_squared_values.append(rsquared)
                    left_indent_sweep.append(left_indent_opt)
                    right_indent_sweep.append(right_indent_opt)
                    sum_width_sweep.append(sum_width_opt)
                    valid_alpha_found = True  # Mark that a valid alpha has been found
                    break  # Exit while loop if a valid alpha is found

            # Reset invalid indices lists after each file
            invalid_sum_indices = []
            invalid_left_indices = []
            invalid_right_indices = []



        hf = h5py.File(path + key + ".h5", 'w')
        hf.create_dataset("alpha", data=alphas)
        hf.create_dataset("wavelength", data=wavelengths)
        hf.create_dataset("r_squared", data=r_squared_values)
        hf.create_dataset("alpha_variance", data=alpha_variances)
        hf.create_dataset("left_indent", data=left_indent_sweep)
        hf.create_dataset("right_indent", data=right_indent_sweep)
        hf.create_dataset("sum_width", data=sum_width_sweep)
        hf.close()
