#!/usr/bin/env python

import imp
import os

from distutils.core import setup

name = 'nengolib'
root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, name, 'version.py'))

setup(
    name=name,
    description="Nengo Library",
    version=version_module.version,
    author="Aaron Voelker",
    author_email="arvoelke@gmail.com",
    url="https://github.com/arvoelke/%s" % name,
    packages=[name],
)
