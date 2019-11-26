# Standard library imports ...
import pathlib
import re

# Third party library imports ...
from setuptools import setup

kwargs = {
    'name': 'Glymur',
    'description': 'Tools for accessing JPEG2000 files',
    'long_description': open('README.md').read(),
    'author': 'John Evans',
    'author_email': 'john.g.evans.ne@gmail.com',
    'url': 'https://github.com/quintusdias/glymur',
    'packages': ['glymur', 'glymur.data', 'glymur.lib'],
    'package_data': {'glymur': ['data/*.jp2', 'data/*.j2k', 'data/*.jpx']},
    'entry_points': {
        'console_scripts': ['jp2dump=glymur.command_line:main'],
    },
    'license': 'MIT',
    'test_suite': 'glymur.test',
    'install_requires': ['gdal', 'lxml', 'numpy', 'setuptools'],
}

kwargs['classifiers'] = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: Implementation :: CPython",
    "License :: OSI Approved :: MIT License",
    "Development Status :: 5 - Production/Stable",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows :: Windows XP",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
    "Topic :: Software Development :: Libraries :: Python Modules"
]

# Get the version string.  Cannot do this by importing glymur!
p = pathlib.Path('glymur') / 'version.py'
contents = p.read_text()
pattern = r'''version\s=\s"(?P<version>\d*.\d*.\d*.*)"\s'''
match = re.search(pattern, contents)
kwargs['version'] = match.group('version')

setup(**kwargs)
