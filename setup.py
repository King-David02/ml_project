from setuptools import find_packages,setup
from typing import List

ignore = '-e .'
def get_requirements(file_path:str)->List[str]:

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        if ignore in requirements:
            requirements.remove(ignore)
    return requirements


setup(
name="my_ml_project",
version="0.0.1",
author='King-David Ajana',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)
