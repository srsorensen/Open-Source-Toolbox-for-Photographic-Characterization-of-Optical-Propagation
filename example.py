from PIL import Image
import sys
sys.path.append("C:/Users/shd-PhotonicLab/PycharmProjects/SPA")
from SPA import SPA
import numpy as np
import matplotlib.pyplot as plt


path = "C:/Users/simon/PycharmProjects/SPA/wavelength_sweep_IRFilter_945nm_optimized_ST3_width_1350nm_TM_1_935.4nm.bmp"
image = Image.open(path)


spa = SPA(True,4870) #set flag to False to turn off plotting

image = spa.rotate_image(image,"flip")

#spa.manual_input_and_output(input_point, output_point)
#spa.set_um_per_pixel(input_point,output_point)

left_indent = 200
right_indent = 200
waveguide_sum_width = 80
IQR_neighbor_removal = 5
sum_width = 80

left_indent_opt = spa.optimize_parameter("left indent", image, left_indent, right_indent, waveguide_sum_width, IQR_neighbor_removal)
right_indent_opt = spa.optimize_parameter("right indent", image, left_indent_opt,right_indent, waveguide_sum_width ,IQR_neighbor_removal)
sum_width_opt = spa.optimize_parameter("sum width", image, left_indent_opt,right_indent_opt, waveguide_sum_width ,IQR_neighbor_removal)
print("The optimal left indent is: ", left_indent_opt, " The optimal right indent is: ", right_indent_opt, "and the optimal sum width is: ", sum_width_opt)
print(spa.analyze_image(image,left_indent_opt,right_indent_opt,sum_width_opt,IQR_neighbor_removal))
