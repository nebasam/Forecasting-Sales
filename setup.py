
#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['pandas>=1.1.0', 'numpy>=1.19.0', 'sklearn',
                'seaborn', 'matplotlib', 'dvc', 'mlflow']

test_requirements = ['pytest>=3', ]

setup(
    author="Nebiyu Samuel",
    email="neba.samuel17@gmail.com",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Machine learning approcah for rossman sales prediction",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='scripts,data exploration, random forest, linear regression, LSTM, RNN',
    name='pharmaceutical-sales-Forecasting',
    packages=find_packages(include=['scripts', 'scripts.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nebasam/Forecasting-Sales',
    version='0.1.0',
    zip_safe=False,
)