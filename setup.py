from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements/requirements.txt").readlines()]


setup(
    name="powerpaint",
    version="0.1.1",
    install_requires=REQUIREMENTS,
    packages=["powerpaint"],
    package_dir= {
        "powerpaint": "powerpaint",
    },
    url="https://github.com/scenario-labs/PowerPaint"
)
