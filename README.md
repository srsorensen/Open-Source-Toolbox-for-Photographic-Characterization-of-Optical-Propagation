This repository contains the Python script SPA.py for scattering analysis of both straight and spiral waveguides. The toolbox is compatible with any USB camera driver and includes the script to control the camera as well.

An example can be seen in the example.py file for both waveguide structures. The scattering analysis is all contained within the SPA class and has been created for a laser input on the left. 
Images can be rotated using the spa.rotate_image function.

The spa.analyse_image function requires four hyperparameters, the left/right indentation, num_neighbors and rows for straight waveguides or simply left and right for spiral waveguides. These can either be determined automatically or be manually input.

The optimal value of the given hyperparameter is determined by running the image analysis for different values of the parameter and subsequently finding regions with the smallest variations in the differentiated data.
