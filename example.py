from PIL import Image
from SPA import SPA
import os


spa_directory = os.getcwd()
spa = SPA(True,4870) #set flag to False to turn off plotting

#Load spiral image
image_path = spa_directory + "/" + "spiral_sample_data.png"

#Distance per pixel
distance_per_pixel = 1.399 # µm

#Bool used to indicate whether the script should optimize parameters or not.
optimize_parameter = False

#Fitting to spiral
alpha_dB_outlier, alpha_dB_outlier_variance, r_squared_outlier, alpha_dB_raw, alpha_dB_raw_variance,r_squared_raw = spa.spiral_waveguide(image_path,distance_per_pixel,optimize_parameter)

#Load straight waveguide
image_path = spa_directory + "/" + "wavelength_sweep_IRFilter_945nm_optimized_ST3_width_1350nm_TM_1_928.7nm.bmp"
image = Image.open(image_path)

#Flip to make sure input is on the left
image = spa.rotate_image(image,"flip")

alpha_dB, r_squared, alpha_dB_variance = spa.straight_waveguide(image,optimize_parameter
