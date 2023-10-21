# Dentate Spikes Classifier

The code in this directory demonstrates how to predict the type of dentate spikes (i.e., either DS1 or DS2) from the local field potential (LFP) signal recorded from the granule cell layer of the dentate gyrus (DG). This work is highlighted in our latest preprint, McHugh et al. 2023.
![Dentate Spike Classifier](./Figures/dSpikesClassifier_schematic.png)
This [notebook](dentateSpikeType_prediction.ipynb) illustrates how to use `dSpikesClassifier` to preprocess the LFP signal and classify dentate spikes into either DS1 or DS during an exemplar session.

The data was acquired using a 64-channel silicon probe spanning the somato-dendritic axis of CA1 principal cells and reaching the inferior blade of the DG. LFP signals were sampled at 1250 Hz. Due to storage constraints of this repository, we have uploaded only the LFPs from the granule cell layer channel of the DG and a channel in the stratum oriens, which is used as a reference for detecting dentate spikes. The data is stored in the '/Data/' directory.

Please note that the data shared here is copyrighted to the University of Oxford.

## Model Information
In order to differentiate between DS1 and DS2 events using solely the LFP traces, we implemented a linear discriminant analysis (LDA) classifier. This classifier was trained using LFP data recorded from the granule cell layer via a silicon probe.

The LFP signals underwent preprocessing, which included down-sampling to 1250 Hz and applying a low-pass filter with a 50 Hz cutoff. We then extracted 400 ms epochs, each centered around the peak of a DS event, spanning from -200 ms to +200 ms with a bin width of 0.8 ms. This resulted in 500 time-based features (dimensions) for each LFP trace, with one feature for each time bin.

To reduce the dimensionality of the data, we conducted principal component analysis (PCA) on all LFP traces related to DS events recorded via the silicon probe, totaling 15,067 traces. This allowed us to extract 16 principal components that collectively explained 90% of the variance in the data. These 16 principal components were subsequently employed to train the LDA classifier.

For more details, please consult our preprint.

## Getting Started

This repository contains the Dentate Spike Classifier, and to ensure a consistent development environment, we recommend creating a Conda virtual environment using the provided `environment.yml` file.

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) should be installed on your system.

### Creating the 'dSpikes' Virtual Environment
1. Open a terminal or command prompt.
2. Navigate to the repository's root directory.
3. Create a new Conda virtual environment called "dSpikes" from the `environment.yml` file using the following command:
``` shell
conda env create -f environment.yml
```
4. Activate the virtual environment:
``` shell
conda activate dSpikes
```

