#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for pysimfrac 
You can install pysimfrac with
python setup.py install 
or
python setup.py install --user 
"""

import os
import sys
import shutil
import datetime
from setuptools import setup

dirs = ["build", "welldetective.egg-info", "dist"]
for d in dirs:
    if os.path.exists(d):
        shutil.rmtree(d)

if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'")
    print()

if sys.version_info[:2] < (3, 8):
    error = """WellDetective requires Python 3.8 or later (%d.%d detected).
""" % sys.version_info[:2]
    sys.stderr.write(error + "\n")
    sys.exit(1)

now = datetime.datetime.now()

name = "WellDetective"

date = now.strftime("%Y-%m-%d %H:%M")

version = "2.0.2"

description = "Analysis of unidentified oil and gas well datasets"

long_description = \
    """
WellDetective is a Python module to analyze unidentified oil and gas well datasets
"""

authors = {
    'Guiltinan': ('Eric Guiltinan', 'eric.guiltinan@lanl.gov'),
    'Taylor': ('Nash Taylor', 'nashctay@lanl.gov'),
    'Lee': ('James E. Lee', 'jamesedlee@lanl.gov'),
    'Santos': ('Javier Santos', 'jesantos@lanl.gov')
}

license = "GPL"

maintainer = "WellDetective Developers, James E. Lee"

maintainer_email = "jamesedlee@lanl.gov"

url = 'https://github.com/lanl/WellDetective/tree/main'

platforms = ['Linux', 'Mac OSX', 'Unix']

keywords = [
    'Orphan Well', 'unidentified orphan well', 'magnetometer', 'methane'
]

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers', 'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Geoscience',
    'Topic :: Scientific/Engineering :: Physics'
]

packages = [
    "WellDetective", "WellDetective.src", "WellDetective.src.general",
    
]

install_requires = ["numpy", "scipy", "matplotlib", 
                    "seaborn", "scikit-gstat", "vedo"]

#from WellDetective import release

if __name__ == "__main__":

    setup(
        name=name.lower(),
        version=version,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        author=authors['Guiltinan'][0],
        author_email=authors['Guiltinan'][1],
        description=description,
        keywords=keywords,
        long_description=long_description,
        license=license,
        platforms=platforms,
        #url=release.url,
        #project_urls=release.project_urls,
        classifiers=classifiers,
        packages=packages,
        install_requires=install_requires,
        python_requires='>=3.8',
        test_suite='nose.collector',
        tests_require=['nose>=1.3.7'],
        zip_safe=False)
