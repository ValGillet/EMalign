from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Align image tiles acquired with SBEM into an image stack. Using SOFIMA (by Google)'
LONG_DESCRIPTION = ''

# Setting up
setup(
        name="emalign", 
        version=VERSION,
        author="Valentin Gillet",
        author_email="valentin.gillet@biol.lu.se",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'tensorstore'], 
        keywords=['python', 'alignment']
    )
