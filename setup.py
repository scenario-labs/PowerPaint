from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]


setup(
    name="powerpaint",
    version="0.1.1",
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    url="https://github.com/scenario-labs/PowerPaint"
)
