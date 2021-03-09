# UtilityBasedRegression

Reference: https://github.com/durgaprasad1997/DynamicRelevanceForRareEvents

1) Install GNU fortran compiler

sudo apt update

sudo apt install gfortran-9

sudo ln -s /usr/bin/gfortran-9 /usr/bin/gfortran

2) Generate fortran python wrapper

python -m numpy.f2py phi.f90 -m phif90_pwrapper -h phi.pyf

python -m numpy.f2py -c phi.pyf phi.f90
