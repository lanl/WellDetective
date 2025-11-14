import datetime

now = datetime.datetime.now()

name = "WellDetective"

date = now.strftime("%Y-%m-%d %H:%M")

version = "1.0"

description = "Analysis of unidentified oil and gas well datasets"

long_description = \
    """
WellDetective is a Python module to analyze unidentified oil and gas well datasets
"""

authors = {
    'Guiltinan': ('Eric Guiltinan', 'eric.guiltinan@lanl.gov'),
    'Santos': ('Javier Santos', 'jesantos@lanl.gov')
}

license = "GPL"

maintainer = "WellDetective Developers, Eric Guiltinan"

maintainer_email = "eric.guiltinan@lanl.gov"

url = 'https://github.com/Eric-Guiltinan/WellDetective'

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
