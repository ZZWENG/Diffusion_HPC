import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output

class TRT_Runtime(object):
    def __init__(self) -> None:
        f = open('/home/yusun/ROMP/trained_models/ROMP.trt', 'rb')
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
    
    def __call__(self, input_batch):
        # allocate device memory
        d_input = cuda.mem_alloc(1 * input_batch.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)

        bindings = [int(d_input), int(d_output)]

        stream = cuda.Stream()