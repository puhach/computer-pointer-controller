from openvino.inference_engine import IENetwork, IECore
from collections import deque, Counter
import cv2
import numpy as np
import logging

class GenericModel:

    core = IECore()

    """
    The parent class for various object detection and recognition models.
    """
    def __init__(self, model_name, concurrency=0, device='CPU', extensions=None):
        """
        Initializes the generic model.
        """

        # OpenVINO 2020.1 loads extensions automatically
        if extensions: #and device.lower() == 'cpu':
            self.core.add_extension(extension_path=extensions, device_name=device)


        self.model_xml = model_name + '.xml'
        self.model_bin = model_name + '.bin'

        #self.network = self.core.read_network(model=self.model_xml, weights=self.model_bin)
        self.network = IENetwork(model=self.model_xml, weights=self.model_bin)

        # Check for unsupported layers
        supported_layers = set(self.core.query_network(network=self.network, device_name=device).keys())
        net_layers = set(self.network.layers.keys())
        unsupported_layers = net_layers.difference(supported_layers)
        if unsupported_layers:
            raise Exception('Unsupported layers: ' + ','.join(unsupported_layers))


        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))

        self.exe_network = self.core.load_network(network=self.network, device_name=device, num_requests=max(1, concurrency))

        self.concurrency = concurrency
        self.request_index = 0
        self.waiting_queue = deque()
        self.output_queue = deque()
        self.stats = Counter()


    def feed_input(self, image):
        """
        Creates the input dictionary from the input image and feeds it to the model for inference. 
        """
        if image is None:
            input_dict = None
        else:
            input_dict = { self.input_name : image }
        self.feed_input_dict(input_dict)


    def feed_input_dict(self, input_dict):
        """
        Takes in the input dictionary and feeds it to the model for inference.
        If the inference mode is synchronous or the number of asynchronous requests
        exceed the limit, the call will block until at least one request is completed.
        """

        if self.concurrency == 0:   # Synchronous inference            
            if input_dict:
                self.exe_network.requests[0].infer(input_dict)
                self.output_queue.append(self.exe_network.requests[0].outputs)
                self._augment_perf_counts(self.exe_network.requests[0].get_perf_counts())
            else:
                self.output_queue.append(None)
        
        else:   # Asyncronous inference
            
            # If we hit the limit of concurrent requests, wait until something is finished
            if len(self.waiting_queue) >= self.concurrency:
                cur_request = self.waiting_queue.popleft()
                # However, the waiting queue may contain requests, which are None (e.g. bad input indicators).
                # We can remove a None request from the queue (so it's no longer overloaded) and pick the next
                # free request (there must be one since we shortened the waiting queue and the number of busy 
                # requests is always less than or equal to the length of the queue).
                if cur_request:
                    output_dict = cur_request.outputs if cur_request.wait(-1)==0 else None
                    self._augment_perf_counts(cur_request.get_perf_counts())
                else:
                    output_dict = None
                self.output_queue.append(output_dict)                
            else:
                cur_request = None

            if not cur_request:
                cur_request = self.exe_network.requests[self.request_index]
                self.request_index = (self.request_index + 1) % self.concurrency


            if input_dict:  
                cur_request.async_infer(input_dict)
                self.waiting_queue.append(cur_request)
            else:
                # Requests for previous frames might be still running,
                # so we can't produce output before they are done
                self.waiting_queue.append(None)


    def consume_output_dict(self, wait):
        """
        Retrieves the output dictionary from the inference results. Returns a tuple. 
        The first value indicates whether the result was retrieved. The second item
        is the output dictionary. The wait parameter specifies whether the function 
        has to wait for the current inference request to finish in case no result is 
        available at the moment of the call. The function never waits if there are no
        busy requests or the result is already available.
        """

        if self.output_queue:
            return True, self.output_queue.popleft()

        if self.waiting_queue:
            request = self.waiting_queue[0]
            if request:
                if wait:    
                    status = request.wait(-1)      
                    outputs = request.outputs if status==0 else None
                    self.waiting_queue.popleft()                    
                    self._augment_perf_counts(request.get_perf_counts())              
                    return True, outputs
                else:   # don't wait                    
                    if request.wait(0) == 0:    # result available
                        self.waiting_queue.popleft()
                        self._augment_perf_counts(request.get_perf_counts())
                        return True, request.outputs
                    else:
                        return False, None
            else:   # request is None
                return True, None

        # Both queues are empty, nothing to wait for
        return False, None  

    def consume_output(self, wait):
        """
        Extracts the inference results from the inference output dictionary.
        Returns a tuple. The first value indicates whether the result was retrieved.
        The second value is the inference result (depends on the model used). 
        The wait parameter specifies whether the function should wait for the 
        current inference request to finish if no result is available.
        """
        consumed, output_dict = self.consume_output_dict(wait)
        output = output_dict[self.output_name] if output_dict else None
        return consumed, output


    def get_stats(self):
        """
        Returns the dictionary containing layer-wise performance counters.
        """
        return self.stats

    def reset_stats(self):
        """
        Resets layer performance counters to zero.
        """
        self.stats = Counter()

    def print_stats(self, title):
        """
        Prints layer-wise performance of the model.
        """
        
        # No need to show the log level and datetime for a multiline message
        logger = logging.getLogger()
        original_formatter = logger.handlers[0].formatter        
        logger.handlers[0].setFormatter(logging.Formatter(fmt='%(message)s'))        
        logger.info(title)

        if self.stats:
            for layer_name, execution_time in self.stats.items():
                # cpu_time and real_time seem to be measured in microseconds, 
                # but the documentation is not really clear about it
                logger.info(f'{layer_name} : {execution_time} us')    
        else:
            logger.info('No data')

        logger.handlers[0].setFormatter(original_formatter) # restore the log format


    def _preprocess_input(self, image, width, height):
        """
        Performs default preprocessing of the input:
        1) Resizes the image to the desired resolution
        2) Converts the channel order from HWC to CHW
        3) Adds the batch dimension
        Protected method, to be used by GenericModel and descendant classes.
        """

        if image is not None and image.size>0 and width>0 and height>0:
            frame_resized = cv2.resize(src=image, dsize=(width, height))
            frame_resized = np.moveaxis(frame_resized, -1, 0)[None,...] # HWC -> BCHW
            return frame_resized
        else:
            return None
    

    def _augment_perf_counts(self, perf_counts):
        """
        Accumulates performance counters for each layer in the model.
        Private method, to be used by GenericModel only.
        """
        layer_timings = { layer_name : v.get('real_time', 0) for layer_name, v in perf_counts.items() }
        self.stats.update(layer_timings)