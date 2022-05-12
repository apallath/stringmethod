from setuptools import setup, find_packages
import pathlib
import pkg_resources

version = "0.0.1"

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    requirements = [
        str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(name='stringmethod',
      version=version,
      description=(
          """Implementation of string method to compute the minimum energy path between
          two points in a multidimensional energy landscape."""
      ),
      long_description=readme,
      long_description_content_type='text/markdown',
      author='Akash Pallath',
      author_email='apallath@seas.upenn.edu',
      url='https://github.com/apallath/stringmethod',
      python_requires='>=3.6',
      install_requires=requirements,
      packages=find_packages())
