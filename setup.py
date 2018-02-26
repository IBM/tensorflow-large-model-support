from setuptools import find_packages
from setuptools import setup

install_requires = [
    'tensorflow >= 1.5',
]

setup(
    name='tflms',
    version='0.1.0',
    description='tflms: graph editing library for large model support',
    author='Tung D. Le',
    author_email='tung@jp.ibm.com',
    packages=find_packages(),
    install_requires=install_requires
)
