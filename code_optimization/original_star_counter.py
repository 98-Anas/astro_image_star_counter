# Original MPI star counter code from documentation
# This code uses MPI and OpenCV to distribute rows of an image and count stars.

import numpy as np
import cv2
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

c = np.array([0])
local_x = np.array([0])
a = np.array((12788,40000),dtype='uint8')

if rank == 0:
        t1 = MPI.Wtime()
        img = cv2.imread("heic1502a.tif",0)
        t2 = MPI.Wtime()
        print " Time taken to open and read the image is : %r sec " %(t2-t1)
        a = np.array(img)

w1 = MPI.Wtime()
remainder = 12788 % size

if rank < remainder:
        rowsize = 12788/size
        rowsize = rowsize + 1
else:
        rowsize = 12788/size

c = np.array((rowsize,40000))
comm.Bcast(c, root=0)
local_x = np.zeros(c,dtype='uint8')
comm.Bcast(local_x, root=0)
total = np.array([0])

comm.Scatterv(a,local_x,root=0)

img_node_thresh = cv2.adaptiveThreshold(local_x,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,59,0)
star_count_node = ((200 < img_node_thresh)).sum()
print " Star count at Rank", rank,"is ", star_count_node

comm.Reduce(star_count_node,total,op=MPI.SUM,root=0)

if comm.rank == 0:
        w2 = MPI.Wtime()
        print " Total Stars ", total
        print " Total time taken", w2-w1 ,"sec"
