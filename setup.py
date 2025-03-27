from setuptools import setup, find_packages

setup(
    name='generalized_contrastive_PCA',
    version='1.0.2',
    author='Eliezyer de Oliveira, Lucas Sjulson',
    author_email='eliezyer.deoliveira@gmail.com',
    description='Python implementation of generalized contrastive PCA methods.',
    packages=find_packages(include=['generalized_contrastive_PCA']),
    install_requires=[
        'numpy',
        'scipy',
        'numba',
    ],
    url='https://github.com/SjulsonLab/generalized_contrastive_PCA',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
