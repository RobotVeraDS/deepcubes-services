from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='deepcubes_services',
    version='0.0.1',
    description='Vera deepcubes services wrappers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='The MIT License',
    url='https://github.com/RobotVeraDS/deepcubes-services',
    author='Robot Vera',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(),

    keywords='nlp',
    install_requires=["deepcubes", "flask", "flask_json"],

    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'flake8',
            'isort',
        ]
    }
)
