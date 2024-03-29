# DL-weather-dynamics

This repository contains the code used to generate results in the paper:

Hakim, G. J., and S. Masanam, 2023: Dynamical tests of a deep-learning weather prediction model. arxiv.org/abs/2309.10867

--

Quick start:

0) clone this repo
git clone git@github.com:modons/DL-weather-dynamics.git

1) set up the environment using conda and pip:
conda env create -f DL_weather_dynamics.yml
conda activate dlwd
pip install onnxruntime-gpu
pip install torch

2) download data (model weights, climatological fields, perturbation fields)
- https://www.atmos.washington.edu/~hakim/DL_weather_dynamics/
- move these files to the paths defined in your configuration file (see item 3) and unzip

3) copy the configuration template file and edit options
- cp config_template.yml config.yml
- path_model: absolute path to model onnx files (downloaded in step 2)
- path_mean: absolute path to the climatological mean state files (downloaded in step 2)
- path_input: absolute path to the perturbation input fields (downloaded in step 2)
- path_output: absolute path to the location where you want the output written

4) run an experiment
- e.g., python run_cyclone.py

5) plot the results
- e.g., python plot_paper_cyclone.py

Notes:

Your results may differ from those shown in the paper for some experiments depending on your environment (hardware, libraries, versions of dependencies, etc.). This is most likely for the heating experiment, but there may be small differences for other experiments as well. Results shown in the paper were run on Intel CPU with 64-bit floats.

Advanced:

The ERA mean state and perturbation files downloaded above can be generated by following these general steps and code in the data_prep directory. You will need to install all needed dependencies, establish an account and API key at Copernicus, etc.

1) Download sample GRIB data using the provided scripts. Edit the output_path variable to send the data to your preferred location.
- download_panguweather_loopdates_djf.py
- download_panguweather_loopdates_jas.py

2) Compute the time-mean from the sample data
- compute_mean.py

3) Compute the time-mean tendencies to insure the mean state is a solution of the model
- mean_state_tendency.py

4) Compute the perturbations by linear regression on the sample data
- regression_initial_condition.py
