
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='VisionAPI',
    version='0.1.3',
    author='Jose Benitez',
    author_email='benitez.ing@gmial.com',
    description='VisionAPI - a Python library for GPT-Based Vision Models inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/josebenitezg/VisionAPI',
    packages=['visionapi'],
    include_package_data=False,
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8,<3.12.0',
    install_requires=requirements,
)