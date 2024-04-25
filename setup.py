from setuptools import find_packages, setup

setup(
    name="jjuke-diffusion",
    version="0.0.1.0",
    description="Use diffusions for various models",
    author="JJukE",
    author_email="psj9156@gmail.com",
    url="https://github.com/JJukE/JJukE_Diffusion.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "einops",
        "scipy",
        "jjuke"
    ],
    extras_require={},
    setup_requires=[],
    test_require=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3",
    keywords=["JJukE_Diffusion", "JJukE-Diffusion", "jjuke_diffusion", "jjuke-diffusion"]
)