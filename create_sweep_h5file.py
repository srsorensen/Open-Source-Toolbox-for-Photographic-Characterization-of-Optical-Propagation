# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:20:07 2023

@author: frede
"""
import traceback
from PIL import Image
from SPA import SPA
import os
import h5py
import numpy as np

def get_wavelength(filename):
    return str(np.round(float(filename.split(".bmp")[0].split("_")[-1][:-2]),1))

#Specific directory, file ending and keyword to load desired files
path = 'D:/Top_Down_Method/GaAs_data/'
f_ending = '.bmp'
contains = '1525'



image_path_dict = {}

#Getting and printing all the found file names based on the above criteria.
for image_dir in os.listdir(path):
    if os.path.isdir(path + image_dir) and contains in image_dir:
        filenames = []
        for file in os.listdir(path + image_dir):
            if file.endswith(f_ending):
                filenames.append(file)
        image_path_dict[image_dir] = filenames

print(image_path_dict.keys())

#Initially runs a test data process so make sure everything works as intented.
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
        #Initiate variables for all the desired outputs of the function and the optimization procedure
        files = image_path_dict[key]
        wavelengths = []
        r_squared_values = []
        alphas = []
        alpha_variances = []
        data_x = []
        data_y = []
        right_indent_sweep = []
        left_indent_sweep = []
        sum_width_sweep = []
        for file in files:
            #Open the image and attempt to find the optimal parameters. A try except loop is used as sometimes this fails.
            try:
                image = Image.open(path + key + "/" + file)
                image = spa.rotate_image(image,"flip")
            except:
                print(f"PIL error at: {file}")
                traceback.print_exc()
                continue

            try:
                left_indent_opt = spa.optimize_parameter("left crop",image,left_indent,right_indent,waveguide_sum_width,IQR_neighbor_removal)
                right_indent_opt = spa.optimize_parameter("right crop",image,left_indent,right_indent,waveguide_sum_width,IQR_neighbor_removal)
                sum_width_opt = spa.optimize_parameter("sum width",image,left_indent,right_indent,waveguide_sum_width,IQR_neighbor_removal)
                alpha, rsquared, alpha_variance = spa.analyze_image(image, left_indent_opt, right_indent_opt, sum_width_opt,
                                                          IQR_neighbor_removal)
                print(str(counter)+ ",  Indent: "+ str(left_indent_opt) + ",  Alpha: " +str(np.round(alpha,1)) + ' dB/cm')
                counter = counter + 1
            except:
                traceback.print_exc()
                print("Error")
                continue
            finally:
                image.close()
            #Remove non physical results. This could be expanded to remove very large values which soemtimes also occur, such as 1000 dB/cm
            if alpha <= 0 or np.isinf(alpha) or np.isnan(alpha):
                continue
            else:
                #print(alpha, type(alpha))
                wavelength = get_wavelength(file)
                alphas.append(alpha)
                alpha_variances.append(alpha_variance)
                wavelengths.append(wavelength)
                r_squared_values.append(rsquared)
                left_indent_sweep.append(left_indent_opt)
                #right_indent_sweep.append(right_indent_opt)
                sum_width_sweep.append(sum_width_opt)


        #Save all the obtained variables in a .h5 file
        hf = h5py.File(path + key + ".h5", 'w')
        hf.create_dataset("alpha", data=alphas)
        hf.create_dataset("wavelength", data=wavelengths)
        hf.create_dataset("r_squared", data=r_squared_values)
        hf.create_dataset("alpha_variance", data=alpha_variances)
        hf.create_dataset("left_indent", data=left_indent_sweep)
        hf.create_dataset("right_indent", data=right_indent_sweep)
        hf.create_dataset("sum_width", data=sum_width_sweep)
        hf.close()
