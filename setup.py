#!/usr/bin/env python

import imp
import io
import os
from setuptools import setup, find_packages


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

name = 'nengolib'
root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, name, 'version.py'))

deps = [  # https://github.com/nengo/nengo/issues/508
    "nengo>=2.1.0",
    "numpy>=1.10",
    "scipy>=0.17.0",
]

download_url = (
    'https://github.com/arvoelke/nengolib/archive/v%s.tar.gz' % (
        version_module.version))

setup(
    name=name,
    version=version_module.version,
    author="Aaron R. Voelker",
    author_email="arvoelke@gmail.com",
    description="Tools for robust dynamics in Nengo",
    long_description=read("README.rst", "CHANGES.rst"),
    url="https://github.com/arvoelke/nengolib/",
    download_url=download_url,
    license="Free for non-commercial use (see Nengo license)",
    packages=find_packages(),
    setup_requires=deps,
    install_requires=deps,
    keywords=[
        'Neural Engineering Framework',
        'Nengo',
        'Dynamical Spiking Networks',
        'Neural Dynamics',
        'Reservoir Computing',
    ],
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
