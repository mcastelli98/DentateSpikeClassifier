[![DOI](https://zenodo.org/badge/704126101.svg)](https://zenodo.org/doi/10.5281/zenodo.10034432)

# Dentate Spikes Classifier
The code in this directory demonstrates how to predict the type of dentate spikes (i.e., either DS1 or DS2) from the local field potential (LFP) signal recorded from the granule-cell layer of the dentate gyrus (DG). This work is highlighted in our [article](https://www.cell.com/neuron/fulltext/S0896-6273(24)00646-9), McHugh et al. 2024.

<p align="center">
    <img src="./Figures/dSpikesClassifier_schematic.png" alt="Dentate Spike Classifier" height="300">
</p>

This [notebook](dentateSpikeType_prediction.ipynb) illustrates how to use `dSpikesClassifier` to preprocess the LFP signal and classify dentate spikes into either DS1 or DS during an exemplar session.

A colab notebook version is also available [here](https://colab.research.google.com/drive/1lRk2JoGZzlhHHjxh8sN4BjzhJK1DHHob?usp=sharing)

The data was acquired using a 64-channel silicon probe spanning the somato-dendritic axis of CA1 principal cells and reaching the inferior blade of the DG. LFP signals were sampled at 1250 Hz. Due to storage constraints of this repository, we have uploaded only the LFPs from the granule cell layer channel of the DG and a channel in the stratum oriens, which is used as a reference for detecting dentate spikes. The data is stored in the `/Data/` directory.

Please note that the data shared here is copyrighted to the University of Oxford.

## Model Information
In order to differentiate between DS1 and DS2 events using only the LFP traces, we implemented a linear discriminant analysis (LDA) classifier. This classifier was trained using LFP data recorded from the granule cell layer (GCL) via a silicon probe.

Prior to prediction, the model performs dimensionality reduction by applying principal component analysis (PCA) to all LFP traces centered around DS events. The principal components are then fed into the LDA classifier.

Following a cross-validation (k=1,000), the classifier achieved a median accuracy of $82.0$%  $(81.6-82.5)$% (IQR).

For more details regarding the preprocessing and training of the model, please consult our [article](https://www.cell.com/neuron/fulltext/S0896-6273(24)00646-9).
<p align="center">
    <img src="./Figures/dSpikesModel_preprint.png" alt="Model Information" height="250">
</p>

## Citing This Work
If you use this software in your research, please cite our work using the provided DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14849859.svg)](https://doi.org/10.5281/zenodo.14849859)

## Getting Started
To ensure a consistent development environment, we recommend creating a Conda virtual environment using the following instructions.

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) should be installed on your system.

### Creating the 'dSpikes' Virtual Environment
1. Open a terminal or command prompt.
2. Navigate to the repository's root directory.
3. Create a new Conda virtual environment called "dSpikes" using the following command:
``` shell
conda create --name dSpikes python=3.10 numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```
4. Activate the virtual environment:
``` shell
conda activate dSpikes
```

