#!/usr/bin/python

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
    fp = open(f, 'r')
    all = fp.read()
    fp.close()

    compute_target = 'compute_20'
    if target is not None:
        if "sm_11" in target:
            compute_target = 'compute_11'
        elif "sm_12" in target:
            compute_target = 'compute_12'
        elif "sm_13" in target:
            compute_target = 'compute_13'
        elif "sm_20" in target:
            compute_target = 'compute_20'
        else:
            print "ERROR: Invalid compute target: %s" % target
            sys.exit(-1)

    place = all.find("static void __sti____cudaRegisterAll")
    all = all[:place]+"extern\"C\" {void ** __cudaRegisterFatBinary2(void*, size_t);}\n"+all[place:]

    place = all.find("__cudaRegisterFatBinary((void*)&__fatDeviceText);")
    version = "3.1"
    if place == -1:
        place = all.find("__cudaRegisterFatBinary((void*)(&__fatDeviceText));")
        version = "2.2"
    print "the place is", place

    if version == "3.1":
        all = all[0:place+len("__cudaRegisterFatBinary")] + "2((void*)(&__fatDeviceText), sizeof(__deviceText_$" + compute_target + "$));" + all[place+len("__cudaRegisterFatBinary((void*)(&__fatDeviceText)"):]
    elif version == "2.2":
        all = all[0:place+len("__cudaRegisterFatBinary")] + "2((void*)(&__fatDeviceText), sizeof(__deviceText_$" + compute_target + "$)" + all[place+len("__cudaRegisterFatBinary((void*)(&__fatDeviceText)"):]

    print "change will be:"
    print all[place-100:place+200]

    fp = open(f, 'w')
    fp.write(all)
    fp.close()

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
