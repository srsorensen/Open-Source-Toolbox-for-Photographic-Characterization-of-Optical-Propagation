from PIL import Image
import sys
sys.path.append("C:/Users/shd-PhotonicLab/PycharmProjects/SPA")
from SPA import SPA
import numpy as np
import matplotlib.pyplot as plt


path = "D:/Top_Down_Method/2023-12-06_wavelength_sweep_IRFilter_945nm_optimized_1_ST3_width_1350nm_TM_2/wavelength_sweep_IRFilter_945nm_optimized_1_ST3_width_1350nm_TM_2_926.7nm.bmp"
image = Image.open(path)


spa = SPA(True,4870) #set flag to False to turn off plotting

image = spa.rotate_image(image,"flip")

#spa.manual_input_and_output(input_point, output_point)
#spa.set_um_per_pixel(input_point,output_point)

left_indent = 200
right_indent = 105
waveguide_sum_width = 80
IQR_neighbor_removal = 5
sum_width = 80

right_indent_opt = spa.find_optimal_right_indent(image, left_indent,waveguide_sum_width ,IQR_neighbor_removal)
left_indent_opt = spa.find_optimal_left_indent(image, right_indent_opt ,waveguide_sum_width ,IQR_neighbor_removal)
sum_width_opt = spa.find_optimal_waveguide_sum_width(image, left_indent_opt, right_indent_opt,IQR_neighbor_removal)
print(right_indent_opt)
print(left_indent_opt)
#print("The optimal left indent is: ", left_indent_opt, " The optimal right indent is: ", right_indent_opt, "and the optimal sum width is: ", sum_width_opt)
print(spa.analyze_image(image,left_indent_opt,right_indent_opt ,sum_width_opt,IQR_neighbor_removal))
