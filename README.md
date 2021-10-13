# qopt-applications
Applications of the [qopt](https://github.com/qutech/qopt) software for quantum
simulation and optimal control.

If you have any questions, comments or recommendations, please reach out to me
via mail at j.teske@fz-juelich.de or julian.teske@rwth-aachen.de.

## Documentation
The documentation of qopt can be found on 
[readthedocs](https://qopt.readthedocs.io/en/latest/index.html). 
It features an API documentation and an introduction in the 
form of jupyter notebooks demonstrating how to utilize the package. A 
complementary theoretical introduction is given in the
[qopt paper](https://arxiv.org/abs/2110.05873).

## Purpose
Publishing the sourcecode of your research has multiple benefits to you and the
scientific community as whole. You can increase the visibility of your 
work and increase the trust in your results by proving the reproducibility of 
your results. New researches entering the field can use your code as starting 
point for their own work, making it very likely that they will read and cite 
your publications.

The scientific community can advance faster as whole if cooperation is easy
and widely applied. The use of qopt as shared platform and the ability to 
share the simulation code in a repository like this one facilitates the 
exchange of ideas and techniques.


## Format Guideline
These guidelines should be seen as recommendations. If they cannot be applied
to your sourcecode then you should rather upload your code in another structure
than not sharing it at all. If you are unsure, then please contact me.

The guidelines are meant to facilitate the use of previous simulation code.
They are set up as compromise between the effort made to document code while
it is in use, and the effort to reuse or refactor old code.

Each project is stored in a separate folder, which contains a README, a SETUP,
an execution script and further code in a subfolder.

### README
This readme should contain a brief description of the physical contend of your
simulation, such that others will know what is simulated. Then it also explains 
the structure of your simulations and any additional information you want to 
provide to fellow researchers (like citations, yours and the ones you used). 

### SETUP
This file should be a python (.py) file containing the setup of your 
simulations. It stores parameters, Hamiltonians, construction functions for 
qopt classes, additional simulation code etc. If your project is too large for 
a single file then add a folder for the files required for the setup. This file 
should be well documented following the numpy docstring format, as it is the
core of your simulations, and the part that will be refactored the first.

### Execution
This file is preferably an ipython notebook which executes the key simulation
using the setup.py file. It does not need to cover your full investigations but
just the most relevant ones, which are probable to be refactored in the future.
This code should be well documented in the numpy format.

### Further Code
This folder contains the parts of your code that are more specific to your own 
work. It should contain another README file explaining the structure, contend
and interdependence of the remaining files. It is sufficient to write short 
document strings in these files, which describe the code roughly, but you are 
welcome to put more effort into the documentation if you deem it useful.

## Citing

If you are using qopt for your work then please cite the 
[qopt paper](https://arxiv.org/abs/2110.05873), as the funding of the 
development depends on the public impact.
