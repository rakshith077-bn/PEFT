from setuptools import setup, find_packages

setup(
        name='peft',
        version='1.0',
        packages=find_packages(),
        install_requires=[
            pandas,
            filelock,
            numpy,
            peft,
            pillow,
            python-dateutil,
            tokenizers,
            torch,
            torchvision=='0.20.1',
            tqdm,
            transformers=='4.46.3',
            scikit-learn,
            lion-pytorch,
            typer,
            pyfiglet,
            progress
        ],

        extras_require={python_requires=="3.10`"}
    )
