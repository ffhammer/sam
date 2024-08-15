from setuptools import setup, find_packages, Extension

# If you use Extension, define it properly as shown in previous examples

setup(
    name='py_lmcurve_ll5',
    version='0.1.0',
    packages=find_packages(),
    author='Felix Hammer',
    author_email='fhammer@uos.de',
    python_requires='>=3.6',
    # Include package data
    package_data={'py_lmcurve_ll5': ['r_lmcurve_ll5.so']},  # Adjust path if necessary
    include_package_data=True,
    ext_modules=[Extension('py_lmcurve_ll5.r_lmcurve_ll5', ['py_lmcurve_ll5/r_lmcurve_ll5.c'])]
)
