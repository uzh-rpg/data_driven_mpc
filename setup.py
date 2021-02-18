from setuptools import setup, find_packages

"""Setup module for project."""

setup(
    name='Data-Driven MPC for Quadrotors',
    version='0.1',
    description='',

    author='Guillem Torrente i Marti',
    author_email='guillemtorrente@hotmail.com',

    packages=find_packages(exclude=[]),
    python_requires='==3.6',
    install_requires=[
        'numpy==1.19.0',
        'scipy==1.5.0',
        'tqdm==4.46.1',
        'matplotlib==3.2.2',
        'scikit-learn==0.23.2',
        'casadi==3.5.1',
        'pyquaternion==0.9.5',
        'joblib==0.15.1',
        'pandas==1.0.5',
        'PyYAML==5.3.1',
        'pycryptodomex==3.9.8',
        'gnupg==2.3.1',
        'rospkg==1.2.8',
        'tikzplotlib==0.9.4'
    ],
)
