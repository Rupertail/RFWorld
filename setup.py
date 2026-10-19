import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.lstrip().startswith("#")
    ]

# Installs
setuptools.setup(
    name="rfworld",
    version="1.0.0",
    author="Junzhe Wang, Jinqing Zhang, Qingjie Liu",
    author_email="qingjie.liu@buaa.edu.cn",
    description="RFWorld end-to-end driving agent for NAVSIM 1.0",
    url="https://github.com/Rupertail/RFWorld",
    python_requires=">=3.9",
    packages=setuptools.find_namespace_packages(where=script_folder, include=["navsim*"]),
    package_dir={"": "."},
    include_package_data=True,
    package_data={"navsim": ["**/*.yaml", "**/*.yml"]},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
