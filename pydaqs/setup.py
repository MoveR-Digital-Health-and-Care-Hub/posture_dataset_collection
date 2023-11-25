# License: GPLv3

from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import pydaqs

VERSION = pydaqs.__version__
PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')

setup(name = "pydaqs",
      version=VERSION,
      description = "DAQ wrappers for axopy.",
      author = "Agamemnon Krasoulis",
      author_email = "agamemnon.krasoulis@gmail.com",
      url = "https://github.com/agamemnonc/pydaqs",
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt']
                    },
      include_package_data=True,
      install_requires=install_reqs,
      license='GPLv3',
      platforms='any',
      long_description="""
A collection of wrapper functions for DAQ packages and libraries in Python.
Mainly intended for internal use within IntellSensing Lab.
""")
