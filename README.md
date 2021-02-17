
# Zeke W. MSc Project Code
This repo contains the code library for the paper "Burst Coding Despite Unimodal Interval Distributions" by Williams et al.

## Folders

### Results
Folder containing data required for plots. Labeled by figure number from paper.

##Files

### plot_figure3-6:
Code for plotting the associated figure. Each of these require the "results" folder to be in the current workspace and some of these require extra libraries contained in this repo.

### simulate3-6
Code to simulate model and run analysis required for the associated figure of the paper. Requires cell_params3-6, cell_models and main_functions libraries. Note that simulate5 has suffix 1, 2 and simulate6 has suffix BSRM and SRM because two different models were used in the generation of each of these figures.

**Note: For the given simulation to run you must change the parameters in the file lba_params so that they match the values used, in the paper, for the given figure**. The default values are for figure 3.

### main_functions
This library contains the core functions used to implement the decoding cell machinery and Stein's linear information rate analysis.

Uses cell params and lba_params files. **Note: to reproduce a figure you must adjust the lba_params file accordingly (see above)**.

### cell_models
This library contains a Gerstner style SRM model, and BSRM models for doublet bursts or a fixed, arbitrary number of intra-burst spikes. Specifically, this library allows one to simulate a population of any number of uncoupled cells, of the given model, with each cell receiving the same Ornstein-Uhlenbeck input.

Contains code to run (i) SRM0 model (simulator: method0) (ii) BSRM with doublets (simulator: method4) or with an arbitrary, fixed number of intra-burst spikes (simulator: method6).

Uses cell_params file.

### gamma_renewal
Contains code to simulate a stationary gamma renewal process. This is used in figure 6 of the paper.

### analyze_simulated
Contains code for various spiketrain statistics (LV, CV2, etc) and their supporting functions.

### cell_params
Dictionary containing cell params used in cell_models code

### lba_params ("lower bound analysis")
Dictionary containing params used in the model of decoding cell machinery and in Stein's lower bound information rate analysis.





