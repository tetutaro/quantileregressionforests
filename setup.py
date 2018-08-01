#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages
import quantileregressionforests


def setup_package():
    metadata = dict()
    metadata['name'] = quantileregressionforests.__package__
    metadata['version'] = quantileregressionforests.__version__
    metadata['description'] = quantileregressionforests.description_
    metadata['author'] = quantileregressionforests.author_
    metadata['url'] = quantileregressionforests.url_
    metadata['license'] = 'MIT'
    metadata['packages'] = find_packages()
    metadata['include_package_data'] = False
    metadata['install_requires'] = [
        'scikit-learn',
    ]
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
