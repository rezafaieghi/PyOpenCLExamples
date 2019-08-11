import pyopencl as cl
import numpy as np

"""Set up OpenCL infrastructure"""
platform = cl.get_platforms()[1]
device = platform.get_devices()
context = cl.Context(device)
queue = cl.CommandQueue(context, device[0], cl.command_queue_properties.PROFILING_ENABLE)

#device.get_info(cl.device_info.LOCAL_MEM_SIZE)

"""Set up GPU program"""
program_file = open('reduction.cl', 'r')
program_text = program_file.read()
program = cl.Program(context, program_text)
try:
    program.build()
except:
    print("Build log:")
    print(program.get_build_info(device, cl.program_build_info.LOG))
    raise
reduction_vector_kernel = cl.Kernel(program, 'reduction_vector')
reduction_complete_kernel = cl.Kernel(program, 'reduction_complete')
program_file.close()


"""Set up kernel data exchange piple"""


data_size = np.ones(1, dtype = np.int32) * 1024
data = np.ones(data_size, dtype = np.float32)
sum_data = np.zeros(1, dtype = np.float32)
actual_sum = np.zeros(1, dtype = np.float32)

data_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostBuf = data)
sum_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostBuf = sum_data)
global_size = data_size / 4
local_size = 32

reduction_vector_kernel.set_args(0, data_buffer)
reduction_vector_kernel.set_args(1, cl.LocalMemory(local_size * 16))

reduction_complete_kernel.set_args(0, data_buffer)
reduction_complete_kernel.set_args(1, 1, cl.LocalMemory(local_size * 16))
reduction_complete_kernel.set_args(2, sum_buffer)


cl.enqueue_copy(queue, data_buffer, data, is_blocking=True)
cl.enqueue_copy(queue, sum_buffer, sum, is_blocking=True)

cl.enqueue_nd_range_kernel(queue, reduction_vector_kernel, global_size, local_size)
print(global_size)

while(global_size > local_size):
    global_size = global_size / local_size
    cl.enqueue_nd_range_kernel(queue, reduction_vector_kernel, global_size, local_size)
    print(global_size)

global_size = global_size/local_size;
cl.enqueue_nd_range_kernel(queue, reduction_vector_kernel, global_size, local_size)

cl.enqueue_copy(queue, sum, sum_buffer)



"""second method"""
#global_size = data_size / 4
#local_size = 32
#cl.enqueue_copy(queue, data_buffer, data, is_blocking=True)
#cl.enqueue_copy(queue, sum_buffer, sum, is_blocking=True)
#
#reduction_vector_kernel(queue, global_size, local_size, data_buffer, cl.LocalMemory(local_size * 16)) #2nd arg.: global size, 3rd arg.: local size
#print(global_size)
#
#while(global_size > local_size):
#    global_size = global_size / local_size
#    reduction_vector_kernel(queue, global_size, local_size, data_buffer, cl.LocalMemory(local_size * 16))
#    print(global_size)
#
#global_size = global_size/local_size;
#reduction_complete_kernel(queue, global_size, local_size, data_buffer, cl.LocalMemory(local_size * 16), sum_buffer)
#
#cl.enqueue_copy(queue, sum, sum_buffer)
