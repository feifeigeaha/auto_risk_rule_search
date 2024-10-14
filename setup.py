from setuptools import setup, find_packages

setup(
    name='auto_risk_rule_search',
    version='0.1.0',
    author='Ruofei Zhang',
    author_email='ruofeizhang36@gmail.com',
    description='A package for automated risk rule search and optimization',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/feifeigeaha/auto_risk_rule_search',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'graphviz',
        'IPython',
        'scikit-learn'
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
