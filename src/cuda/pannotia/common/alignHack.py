#!/usr/bin/python

import fileinput
import optparse
import os
import sys

def findFile():
    files = os.listdir('.')
    f = ''
    for i in files:
        if i.endswith('.cu.cpp'):
            f = i
    if not f:
        print "Problem finding a .cu.cpp cucppfilename!!!"
        sys.exit(1)
    print "found: ", f
    return f

def editFile(f, target):
    for line in fileinput.input(f, inplace=1):
        if ".align 32" in line:
            # NOTE: This adds a space in the replacement in order to maintain
            # The same overall line length and avoid spilling the PTX text out
            # of the allocated space in the .cu.cpp file
            line = line.replace(".align 32", ".align 5 ")
            print line
        else:
            print line


parser = optparse.OptionParser()
parser.add_option("--target", "-t", action="store", default=None, help="Specify compile target: sm_11, sm_12, sm_13, or sm_20")
parser.add_option("--cppfile", "-f", action="store", default=None, help="Specify the .cu.cpp file to hack up")
(options, args) = parser.parse_args()

cucppfilename = None
if options.cppfile is not None:
    cucppfilename = options.cppfile
else:
    cucppfilename = findFile()
editFile(cucppfilename, options.target)
