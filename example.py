from PIL import Image
from SPA import SPA
import numpy as np
import matplotlib.pyplot as plt

#Load the image folder
path = "E:/Top_Down_Method/2023-12-05_wavelength_sweep_IRFilter_980nm_optimized_ST3_width_1600nm_TM_3/wavelength_sweep_IRFilter_980nm_optimized_ST3_width_1600nm_TM_3_913.0nm.bmp"
image = Image.open(path)

spa = SPA(True,4870) #set flag to False to turn off plotting. Second argument is the chip length.

image = spa.rotate_image(image,"flip") #Flip image to have input on the left


right_indent = 100
waveguide_sum_width = 80
IQR_neighbor_removal = 5

left_indent_opt = spa.find_optimal_left_indent(image, right_indent ,waveguide_sum_width ,IQR_neighbor_removal)
right_indent_opt = spa.find_optimal_right_indent(image, left_indent_opt ,waveguide_sum_width ,IQR_neighbor_removal)
sum_width_opt = spa.find_optimal_waveguide_sum_width(image, left_indent_opt, right_indent_opt,IQR_neighbor_removal)
print("The optimal left indent is: ", left_indent_opt, " The optimal right indent is: ", right_indent_opt, "and the optimal sum width is: ", sum_width_opt)

alpha_dB, r_squared, fit_x, fit_y, alpha_dB_variance = spa.analyze_image(image,left_indent_opt,right_indent_opt,sum_width_opt,IQR_neighbor_removal)

print(f"The propagation loss is {alpha_dB:.1f} \u00B1 {alpha_dB_variance:.1f} dB/cm with a R\u00b2 of {r_squared:1f}")