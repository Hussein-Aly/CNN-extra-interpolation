# CNN extra/interpolation of images
This project is an ML project and contains a simple CNN that is trained to extrapolate and interpolate
an input image.


### Usage
In our case, the simple usage is:
```
python main.py working_config.json
```

### Structure
Having a tree with the files and folders is nice to get an overview.
However, this is sometimes tedious to maintain and omitted.
```
example_project
|- architectures.py
|    Classes and functions for network architectures
|- data_reader.py
|    Extrapolating pixels from images 
|- data_sets.py
|    Dataset classes and dataset helper functions
|- main.py
|    Main file. In this case also includes training and evaluation routines.
|- README.md
|    A readme file containing info on project, example usage, authors, publication references, and dependencies.
|- utils.py
|    Utility functions and classes. In this case contains a plotting function.
|- working_config.json
|     An example configuration file. Can also be done via command line arguments to main.py.
```

### Dependencies
```
|- requirements.txt
```
