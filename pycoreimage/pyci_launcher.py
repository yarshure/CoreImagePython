from __future__ import absolute_import

import os, subprocess, sys
import pip

if __name__ == '__main__':
    cwd = os.path.dirname(os.path.dirname(__file__))
    print('Running demo launcher from', cwd)

    try:
        os.chdir('/')
        import numpy
        import matplotlib
        import pycoreimage

        from pycoreimage import pyci
        print(numpy.__file__)
        print(matplotlib.__file__)
        print(pycoreimage.__file__)

    except Exception as e:
        print(e)

        print('It appears that you are missing packages for this script to run properly.')

        cmd = 'python setup.py develop --user'
        sys.stdout.write('OK to run `{}` from `{}` ? [Y/N] '.format(cmd, cwd))
        answer = raw_input()

        if answer.lower() == 'y':
            output = subprocess.check_output(cmd, cwd=cwd, shell=True)
            print(output)
            print('Done with installation. Run this script again.')
            exit(0)
        else:
            exit(1)

    # Run demo code
    img = os.path.join(cwd, 'pycoreimage', 'resources', 'Food_1.jpg')
    dataset = os.path.join(cwd, 'pycoreimage', 'resources')
    script = os.path.join(cwd, 'pycoreimage', 'pyci_demo.py')
    cmd = 'python {} {} {}'.format(script, img, dataset)
    subprocess.check_call(cmd, cwd=cwd, shell=True)
