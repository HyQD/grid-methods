from setuptools import setup, find_packages

def find_version(path):
    import re
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(
    name="HyQD-grid-lib",
    version=find_version("grid_lib/version.py"),
    packages=find_packages(),
    install_requires=["scipy", "opt_einsum"],
    include_package_data=True,
)
