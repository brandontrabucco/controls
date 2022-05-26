"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow==2.7.2',
    'numpy']


setup(
    name='diffopt',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('diffopt')],
    description='Differentiable Optimal Controls')
