from setuptools import setup, find_packages

setup(
    name="seamless_communication",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fairseq2>=0.2.1",
        "torch>=1.10.0",
        "torchaudio",
        "numpy",
        "typing_extensions",
        "pydub",
        "sentencepiece",
    ],
    python_requires=">=3.8",
)