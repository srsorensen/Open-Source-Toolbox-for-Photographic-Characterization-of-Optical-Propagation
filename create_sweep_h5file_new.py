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


path = 'E:/AlGaAs_Power/Already Processed/test/'
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
        point_mean_sum = []
        point_mean_left = []
        point_mean_right = []
        invalid_left = []
        invalid_right = []
        invalid_sum = []
        for file in files:
            retries = 0  # To track the number of retries

            while retries < 3:  # Attempt up to 3 times
                try:
                    image = Image.open(path + key + "/" + file)
                    image = spa.rotate_image(image, "flip")
                    if retries == 0:
                        # Calculate all indices on the first attempt or when adjusting the sum index
                        sum_index, point_mean_sum, sum_width_opt = spa.optimize_parameter(
                            "sum width", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            point_mean_sum
                        )
                        left_index, point_mean_left, left_indent_opt = spa.optimize_parameter(
                            "left crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            point_mean_left
                        )
                        right_index, point_mean_right, right_indent_opt = spa.optimize_parameter(
                            "right crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            point_mean_right
                        )
                    if retries == 1:
                        # Recalculate only the left index and related values
                        left_index, point_mean_left, left_indent_opt = spa.optimize_parameter(
                            "left crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            point_mean_left
                        )
                    if retries == 2:
                        # Recalculate only the right index and related values
                        right_index, point_mean_right, right_indent_opt = spa.optimize_parameter(
                            "right crop", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal,
                            point_mean_right
                        )


                    # Analyze the image to get the alpha value
                    alpha, rsquared, alpha_variance = spa.analyze_image(
                        image, left_indent_opt, right_indent_opt, sum_width_opt, IQR_neighbor_removal
                    )

                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    traceback.print_exc()
                    break  # Skip to next file if an exception occurs
                finally:
                    image.close()

                # Check if alpha is valid
                if alpha <= 7 or alpha > 120 or np.isinf(alpha) or np.isnan(alpha):

                    # Check point_mean and adjust indices if necessary
                    if retries == 0:
                        invalid_sum.append(sum_index)
                        for index in invalid_sum:
                            point_mean_sum[index] = None
                        count_non_none = len([item for item in point_mean_sum if item is not None])
                        if count_non_none <= 1:
                            retries += 1

                    if retries == 1:
                        invalid_left.append(left_index)
                        for index in invalid_left:
                            point_mean_left[index] = None
                        count_non_none = len([item for item in point_mean_left if item is not None])
                        if count_non_none <= 1:
                            retries += 1

                    if retries == 2:
                        invalid_right.append(right_index)
                        for index in invalid_right:
                            point_mean_right[index] = None
                        count_non_none = len([item for item in point_mean_right if item is not None])
                        if count_non_none <= 1:
                            retries += 1



                        # Retry the current index adjustment
                    if retries == 3:
                        print('No suitable parameters were found for', file)
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
                    break  # Exit while loop if a valid alpha is found

            # Reset invalid indices lists after each file
            point_mean_sum = []
            point_mean_left = []
            point_mean_right = []
            invalid_left = []
            invalid_right = []
            invalid_sum = []



        hf = h5py.File(path + key + ".h5", 'w')
        hf.create_dataset("alpha", data=alphas)
        hf.create_dataset("wavelength", data=wavelengths)
        hf.create_dataset("r_squared", data=r_squared_values)
        hf.create_dataset("alpha_variance", data=alpha_variances)
        hf.create_dataset("left_indent", data=left_indent_sweep)
        hf.create_dataset("right_indent", data=right_indent_sweep)
        hf.create_dataset("sum_width", data=sum_width_sweep)
        hf.close()
