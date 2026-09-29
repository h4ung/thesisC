from setuptools import find_packages, setup

with open("requirements.txt") as f:
    reqs = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cardioformer-ckd",
    version="0.1.0",
    description="Multimodal prognostic model for incident CKD from MIMIC-IV and "
                "MIMIC-IV-ECG, adapting the Cardioformer ECG transformer.",
    author="cardioformer-ckd contributors",
    license="MIT",
    packages=find_packages(
        include=[
            "data_preprocessing*", "data_provider*", "exp*",
            "layers*", "models*", "utils*",
        ]
    ),
    python_requires=">=3.9",
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
