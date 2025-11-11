from setuptools import setup

setup(
    name="street-sign-ocr",
    version="1.0.0",
    install_requires=[
        "setuptools",
        "wheel",
        "numpy>=1.26.0",
        "opencv-python-headless",
        "easyocr>=1.7.0",
        "streamlit>=1.28.0",
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
    ],
)
