from setuptools import setup

# requirements file vs. setup.py: https://stackoverflow.com/a/33685899/4816930
requirements = [
    'tensorflow-gpu',
    'keras',
    'pandas',
    'numpy',
    'h5py',
    'jupyter'
]

setup(
    name='mp-2018',
    author='Maxi Fischer, Willi Gierke, Ajay Kesar, Thomas Kellermeier, Axel Stebner, Daniel Thevessen',
    description='Unsupervised Anomaly Detection: Representation Learning for Predictive Maintenance over Time',
    long_description=open('README.md').read(),
    version='0.0',
    packages=[],
    scripts=[],
    install_requires=requirements,
    url='github.com/KDD-OpenSource/MP-2018',
    license='MIT License',
)
