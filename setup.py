import os
from setuptools import setup, find_packages


if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = '''A toolkit for resolving chemical SMILES from structural diagrams.'''



setup(
    name='reactiondataextractor',
    version='2.0.0',
    author='Damian Wilary',
    author_email='dmw51@cam.ac.uk',
    license='MIT',
    url='https://github.com/dmw51/reactiondataextractor2',
    description='A toolkit for converting chemical reaction schemes into a machine-readable format.',
    keywords='image-mining mining chemistry cheminformatics OCR reaction scheme structure diagram html computer vision science scientific',
    packages=find_packages(),
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=[
        'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        'decimer @ git+https://github.com/dmw51/DECIMER-Image_Transformer.git',
        'tesserocr==2.5.1',
        'scipy==1.9',
        'numpy',
        'scikit-learn',
        'cirpy',
        'opencv-contrib-python==4.5.*'
        
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)

