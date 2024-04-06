from PIL import Image
from SPA import SPA
import os

#Load current directory
spa_directory = os.getcwd()

#Initiate the SPA class.

chip_length = 4870 #μm
show_plot = True

spa = SPA(show_plot,chip_length)

#Load spiral image
image_path = spa_directory + "/" + "spiral_waveguide_sample_data.png"

#Physical chip distance per pixel
distance_per_pixel = 1.399 # µm

#Bool used to indicate if the script (True) should optimize parameters or (False) be user input.
optimize_parameter = True

#Fitting to spiral
alpha_dB_outlier, alpha_dB_outlier_variance, r_squared_outlier, alpha_dB_raw, alpha_dB_raw_variance,r_squared_raw = spa.spiral_waveguide(image_path,distance_per_pixel,optimize_parameter)

#Load straight waveguide
image_path = spa_directory + "/" + "straight_waveguide_sample_data.bmp"
image = Image.open(image_path)

#Flip to make sure the input is on the left
image = spa.rotate_image(image,"flip")

alpha_dB, r_squared, alpha_dB_variance = spa.straight_waveguide(image,optimize_parameter)