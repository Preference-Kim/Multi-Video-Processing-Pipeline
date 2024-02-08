import pathlib
import re
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()  # current path
long_description = (here / "README.md").read_text(encoding="utf-8")  # Get the long description from the README file

def get_version():
    file = here / "libs/__init__.py"
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(), re.M).group(1)

"""setup"""
if __name__ == '__main__':

    # Configure the setup script
    setup(
        name='cv2mp',
        version=get_version(),

        # Include packages from the 'libs' directory
        package_dir={'': 'libs'},
        packages=find_packages(where='libs'),
    )