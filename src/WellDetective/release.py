import datetime

now = datetime.datetime.now()

name = "WellDetective"

date = now.strftime("%Y-%m-%d %H:%M")

version = "2.0"

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
