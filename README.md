**ReactionDataExtractor** is a toolkit for the automatic extraction of data from chemical reaction schemes.

## Features

- Automatic extraction of chemical reaction schemes
- Segmentation of reaction arrows, conditions, diagrams and labels
- Optical recognition of chemical structures and text, parsing of reaction condiitions
- Whole-reaction-scheme recovery and conversion into a machine-readable format
- High-throughput capabilities
- Direct extraction from image files
- PNG, GIF, JPEG, TIFF image format support


# Installation

This section outlines the steps required to install ReactionDataExtractor. The simplest way is to do this through [**conda**](https://docs.conda.io/en/latest). 

###  Installation via Conda

Anaconda Python is a self-contained Python environment that is useful for scientific applications.

First, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), which contains a complete Python distribution alongside the conda package manager.

Next, go to the command line terminal and clone the git repository by typing

    git clone https://github.com/dmw51/reactiondataextractor2 

Update apt and apt-get, then install the following required packages:

    sudo apt-get install gcc g++ libpotrace-dev pkg-config libagg-dev ffmpeg libsm6 libxext6
    sudo apt install libtesseract-dev
    
Inside the cloned directory, create the conda environment by typing
    
    conda env create -f environment.yaml
    
Once this is created, enter this environment with the command

    conda activate rde2
    
Then, install ReactionDataExtractor by typing

    pip install -e .
    
This installs the full version of ReactionDataExtractor framework. Finally, download the model weights from [Google Drive](https://drive.google.com/file/d/1v-gkH7iEPvcqAMTPyoqM973UzmwoNszV/view?usp=sharing) and extract the archive into reactiondataextractor/models.

If you run into problems with underlying Tesseract engine, clone the following repository:

    git clone https://github.com/tesseract-ocr/tessdata

and replace the TESSDATA_PATH value inside reactiondataextractor/configs/config.py to point to this tessdata directory.

# Getting Started

This page gives a introduction on how to quickly get started with ReactionDataExtractor This assumes you already have
ReactionDataExtractor and all dependencies installed.

## Extract from Images
You can run ReactionDataExtractor using our command line interface by typing: 

    >>> python reactiondataextractor/extract.py --path <path> --output_dir <dir>
    
where <i>path</i> is a pth to a single image or a directory of images and <i>dir</i> is a directory where output files will be stored. If a path to a single image file is given, then the output directory is not necessary. In this case, the information will be returned instead.

The output files contain the reaction graph with all detected and recognised objects in the form. Each file has a list of nodes specifying information about each reaction entity, as well as an adjacency dictionary with node connectivity information.
