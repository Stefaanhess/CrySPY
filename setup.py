from setuptools import setup, find_packages


setup(
    name="cryspy",
    find_packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["pyxtal"],
    version="0.9.0",
)