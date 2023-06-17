import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input")
ap.add_argument("-o", "--output", required=False,
	help="path to output")
ap.add_argument("-n", "--blockmarker", required=False,
	help="path to output")





args = vars(ap.parse_args())
print(args["input"])
ipfile = open(args["input"],"r+", errors="ignore")
data=ipfile.read()
ipfile.close()


print(len(data))

splitter=len(data)/2

con1=data[:int(splitter)]
print(con1)

con2=data[int(splitter):]
print(con2)


import random
n = args["blockmarker"]


hashfile1 = open(args["output"]+"hashblock1_"+str(n)+".hbl","w")#write mode 
hashfile1.write(con1) 
hashfile1.close() 


hashfile2 = open(args["output"]+"hashblock2_"+str(n)+".hbl","w")#write mode 
hashfile2.write(con1) 
hashfile2.close() 

hashfile2 = open(args["output"]+"hashblock3_"+str(n)+".hbl","w")#write mode 
hashfile2.write(con1) 
hashfile2.close() 

hashfile2 = open(args["output"]+"hashblock4_"+str(n)+".hbl","w")#write mode 
hashfile2.write(con1) 
hashfile2.close() 

hashfile2 = open(args["output"]+"hashblock5_"+str(n)+".hbl","w")#write mode 
hashfile2.write(con1) 
hashfile2.close() 
# store its reference in the variable file1  
# and "MyFile2.txt" in D:\Text in file2 
#file2 = open(r"D:\Text\MyFile2.txt","w+")
