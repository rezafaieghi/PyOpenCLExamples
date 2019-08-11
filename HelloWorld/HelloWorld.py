import pyopencl as cl
import numpy as np

"""Set up OpenCL infrastructure"""
platform = cl.get_platforms()[1]
device = platform.get_devices()
context = cl.Context(device)
queue = cl.CommandQueue(context, device[0], cl.command_queue_properties.PROFILING_ENABLE)


"""Set up GPU program"""
program_file = open('sum.cl', 'r')
program_text = program_file.read()
program = cl.Program(context, program_text)
try:
    program.build()
except:
    print("Build log:")
    print(program.get_build_info(device, cl.program_build_info.LOG))
    raise
adder = cl.Kernel(program, 'add')
print("Kernel Name:"),
print(adder.get_info(cl.kernel_info.FUNCTION_NAME))
program_file.close()


"""Set up kernel data exchange piple"""

size = 1000
a = np.ones(size, dtype=np.float32)
b = np.ones(size, dtype=np.float32) * 2
c = np.zeros_like(a)

a_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, a.nbytes)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, b.nbytes)
c_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, c.nbytes)


cl.enqueue_copy(queue, a_buffer, a, is_blocking=True)
cl.enqueue_copy(queue, b_buffer, b, is_blocking=True)

adder(queue, a.shape, None, a_buffer, b_buffer, c_buffer) #2nd arg.: global size, 3rd arg.: local size
cl.enqueue_copy(queue, c, c_buffer)
