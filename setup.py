import setuptools

__version__ = "0.0.1"

REPO_NAME = "Tiny_ML_Model"
AUTHOR_USER_NAME = "tusharchaudharryy"
SRC_REPO = "audio_classifier"
AUTHOR_EMAIL = "tusharchaudhary@agrisciese.com"

setuptools.setup(
    name=REPO_NAME,  
    version=__version__,
    author="Tushar Chaudhary",   
    author_email="tusharchaudhary@agrisciese.com",  
    description="A small python package for an Audio Classification App",
    long_description=open("README.md").read(), 
    long_description_content_type="text/markdown",
    url=f"https://github.com/AgriSciense/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/AgriSciense/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[  
        "numpy",
        "scikit-learn",
        "tensorflow",
    ],
    python_requires=">=3.8",
)
