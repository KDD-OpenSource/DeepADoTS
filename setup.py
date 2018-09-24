from setuptools import setup


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name='deep-adots',
    author='Maxi Fischer, Willi Gierke, Ajay Kesar, Thomas Kellermeier, Axel Stebner, Daniel Thevessen',
    description='Unsupervised Anomaly Detection: Representation Learning for Predictive Maintenance over Time',
    long_description=open('README.md').read(),
    version='0.0',
    packages=[],
    scripts=[],
    # Requirements for executing the project (not development)
    install_requires=parse_requirements('requirements.txt'),
    url='github.com/KDD-OpenSource/DeepADoTS',
    license='MIT License',
)
