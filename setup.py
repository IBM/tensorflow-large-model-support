# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2018, 2019. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************
from setuptools import find_packages
from setuptools import setup

install_requires = [
    'tensorflow >= 1.5',
]

setup(
    name='tflms',
    version='0.4.0',
    description='tflms: graph editing library for large model support',
    author='Tung D. Le',
    author_email='tung@jp.ibm.com',
    packages=find_packages(),
    install_requires=install_requires
)
