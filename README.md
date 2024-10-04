
 
# Still in the making


# Introduction 


This software repository contains the scripts and files necessary to reproduce the results of the article "Self-Supervised
Coherence-Based on Cryoseismological Distributed Acoustic Sensing":
10.22541/au.172779667.76811452/v1 (DOI)

The raw data necessary to run the code is available on zenodo: 

[raw link][![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13868934.svg)](https://doi.org/10.5281/zenodo.13868934)

To fascilitate the code execution without the need of denoising the real-world data, additionally the denoised data of experiment 03_accumulation_horizontal is provided on zenodo:


[denoised link]

Parts of the code are built upon the software provided by van den Endet et al. [1].

# Setup



# Description of single .py files

### calculating_cc.py

To run this script for all experiments, you must download the raw data from zenodo [raw link] and denoise the raw DAS data 
for all experiments first. To run this script for experiment 03_accumulation_horizontal one can or download the raw [raw link]
and denoised data from zenodo [denoised link].

Calculates the local waveform coherence and cross correlation between DAS data and co-located seismometer for every experiment
as described in Section 4.1.1 and Section 4.3.
The values are saved in a .csv file named cc_evaluation_id.csv in the respective experiment folder. 


### denoise_synthetic_DAS.py

To run this script, you must first download the raw data from Zenodo.

Denoises the synthetic data in the folders data/synthetic_DAS/from_DAS and data/synthetic_DAS/from_seis as described in 
Section 3.4. The denoised data is stored in the directory: experiments/experiment_name/denoised_synthetic_DAS.


### denoising_DAS.py

To run this script, you must first download the raw data from Zenodo.

Denoises the raw real-world data in the folder data/raw_DAS as described in Section 3.4. The denoised data is saved under 
experiments/experiment_name/denoisedDAS.


### generate_DAS_preprocessed_training_data.py


### generate_seismometer_preprocessed_training_data.py

### generate_synthetic_test_data.py

### main_training.py

### models.py

### plotting_fig4.py

### plotting_fig5.py

### plotting_fig6.py

### plotting_figS1.py

### plotting_figS2.py

### plotting_figS3-S5.py

### plotting_fig6.py

### retraining.py









# References

[1] van den Ende, M., Lior, I., Ampuero, J.-P., Sladen, A., Ferrari, A. ve Richard, C. (2021, 3 Mart). A Self-Supervised Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing data. figshare. doi:10.6084/m9.figshare.14152277.v1