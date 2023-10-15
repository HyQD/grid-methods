from setuptools import setup, find_packages

setup(
    name="HyQD-grid-methods",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["scipy", "opt_einsum"],
    include_package_data=True,
)
