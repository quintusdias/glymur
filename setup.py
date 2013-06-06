from distutils.core import setup

kwargs = {'name': 'Glymur',
          'version': '0.1.6',
          'description': 'Tools for accessing JPEG2000 files',
          'long_description': open('README.md').read(),
          'author': 'John Evans',
          'author_email': 'johnevans938 at gmail dot com',
          'url': 'https://github.com/quintusdias/glymur',
          'packages': ['glymur', 'glymur.test', 'glymur.lib',
                       'glymur.lib.test'],
          'package_data': {'glymur': ['data/*.jp2']},
          'scripts': ['bin/jp2dump'],
          'license': 'LICENSE.txt',
          'platforms': ['darwin']}
clssfrs = ["Programming Language :: Python",
           "Programming Language :: Python :: 2.7",
           "Programming Language :: Python :: 3.3",
           "Programming Language :: Python :: Implementation :: CPython",
           "License :: OSI Approved :: MIT License",
           "Development Status :: 3 - Alpha",
           "Operating System :: MacOS",
           "Operating System :: POSIX :: Linux",
           "Intended Audience :: Science/Research",
           "Intended Audience :: Information Technology",
           "Topic :: Software Development :: Libraries :: Python Modules"]
kwargs['classifiers'] = clssfrs
setup(**kwargs)
