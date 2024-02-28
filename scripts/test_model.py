'''
This file illustrates the use of the waterheaters library to run a test case

It loads the main function from the library and provides as parameters the input excel file and the folder where
the results should be written
'''
import os,sys
# include the main library path (the parent folder) in the path environment variable
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)

# import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
import waterheaters as wh

