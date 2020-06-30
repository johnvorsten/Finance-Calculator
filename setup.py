# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:44:35 2020

@author: z003vrzk
"""

# Python imports
import setuptools


#%%

with open('README.md', 'r') as f:
    long_description = f.read()

short_description="3D graphing of incomes/expense versus risk"

setuptools.setup(
    name="finance-graph", # Replace with your own username
    version="0.0.1",
    author="John Vorsten",
    author_email="vorstenjohn@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnvorsten/Finance-Calculator",
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['scikit-learn>=0.23.1',
                      'scipy',
                      'matplotlib',
                      'pandas']
)