import glob
import os
import gzip
import Image


for f in glob.glob("*.JPEG"):
    
    try: 
        im = Image.open(f)
       
    except IOError:
        new_name = f + '.gz'
        print f
        os.rename(f, new_name)

        
