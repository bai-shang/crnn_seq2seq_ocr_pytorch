import os
import sys

with open('char_std_5990.txt') as fd:
    cvt_lines = fd.readlines()

cvt_dict = {}
for i, line in enumerate(cvt_lines):
    key = i
    value = line.strip()
    cvt_dict[key] = value

if __name__ == "__main__":
   cvt_fpath = sys.argv[1]
   

   with open(cvt_fpath) as fd:
       lines = fd.readlines()


   for line in lines:
       line_split = line.strip().split()
       img_path = line_split[0]
       label = ''
       for i in line_split[1:]:
           label += cvt_dict[int(i)]
       print(img_path, ' ', label)
