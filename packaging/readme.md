#ImbalancedUtilityBasedSampler

This package provide several methods for handling utility-based learning problem in imbalanced learning domain.

Github link
https://github.com/ndai093/UtilityBasedRegression

The package requires a compiled Python Fortran wrapper library. If there is issue importing Python Fortran wrapper library, following steps may help to solve the problem

- Install GNU fortran compiler
sudo apt update

sudo apt install gfortran-9

sudo ln -s /usr/bin/gfortran-9 /usr/bin/gfortran

- Generate fortran python wrapper
python -m numpy.f2py phi.f90 -m phif90_pwrapper -h phi.pyf

python -m numpy.f2py -c phi.pyf phi.f90

- The generated library should be put into PhiRelevance folder