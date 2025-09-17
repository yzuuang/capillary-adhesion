This file lists all changes to the code

v0.0.2
------
* In simulation script
    * extract parameter values into separate INI files
    * allow parmeter sweeps to be specified in INI files.
* In model
    * add free energy of liquid-solid surface, so it shows a difference between hydrophobic and hydrophillic cases
    * add correction terms from mathematicians.
* In solver
    * fix certain convergence problems by eliminating the gradient towards the infeasible zone.
    * fix "blinks" between steps by passing dual variable (lambda). 

v0.0.1
------

* Initial release