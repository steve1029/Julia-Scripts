from mpi4py import MPI
import subprocess as sp
import numpy as np
import time

#hostname, err = sp.Popen("hostname".split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
#hostname = hostname.rstrip("\n")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

print("Hostname: %s, Rank: %d, of %d nodes" % (hostname, rank, size))

n = 1

comm.Barrier()

if rank == 0:
	print("The message has been sent")
	send_data1 = np.random.random(n).astype(np.float32)
	send_data2 = np.random.random(n).astype(np.float32)
	comm.Send(send_data1, dest=1, tag=1)
	comm.Send(send_data2, dest=1, tag=2)

elif rank == 1:
	print("The message has been received")
	recv_data1 = np.zeros(n, dtype=np.float32)
	recv_data2 = np.zeros(n, dtype=np.float32)
	comm.Recv(recv_data1, source=0, tag=1)
	comm.Recv(recv_data2, source=0, tag=2)

comm.Barrier()

if rank == 0:
	print("SEND1", send_data1)
	print("SEND2", send_data2)
if rank == 1:
	print("RECV1", recv_data1)
	print("RECV2", recv_data2)
