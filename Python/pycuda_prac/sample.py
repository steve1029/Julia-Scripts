# from : http://documen.tician.de/pycuda/tutorial.html 
import pycuda.gpuarray as gpuarray
import pycuda.driver as CUDA
import pycuda.autoinit
import numpy

random = numpy.random.randn( 4 , 4 ).astype(numpy.float32)
a_gpu = gpuarray.to_gpu (random)
a_doubled = ( 2 * a_gpu).get ()
print (a_doubled)
print (a_gpu)
