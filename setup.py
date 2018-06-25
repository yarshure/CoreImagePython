"""pycoreimage setup module.

To install pycoreimage, open a terminal window, and
- cd /path/to/this/directory/
- python setup.py --user

To experiment and make changes to the source code:
- cd /path/to/this/directory/
- python setup.py develop --user
"""

# https://packaging.python.org/guides/distributing-packages-using-setuptools/
from setuptools import setup

# Get current __version__
#execfile('pycoreimage/__init__.py')
exec(open('pycoreimage/__init__.py').read())

setup(
    name='pycoreimage',

    version=__version__,

    description='Python bindings for Core Image',

    # FIXME:
    url='https://developer.apple.com/sample-code/wwdc/2016/',

    author='Apple Inc.',

    # https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],

    keywords='CoreImage',

    packages=['pycoreimage'],

    install_requires=['numpy>=1.13.0',
                      'matplotlib>=1.3.1',
                      'pyobjc>=4.2',
                      ],

)
