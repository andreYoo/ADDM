import gzip
import os
import shutil

import nibabel as nib
import pdb
from utils.NII import *


class MINC(NII):

    def __init__(self, filename):
        if filename.endswith(".mnc"):
            with open(filename, 'rb') as f_in:
                with gzip.open(filename + ".gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        # Load the MINC file using niBabel, convert it to Nifti format
        pdb.set_trace()
        minc = nib.load(filename)
        basename = minc.get_filename().split(os.extsep, 1)[0]

        if not os.path.isfile(basename + ".nii"):
            out = nib.Nifti1Image(minc.get_data(), affine=minc.affine)
            nib.save(out, basename + '.nii')

        NII.__init__(self, basename + '.nii')
