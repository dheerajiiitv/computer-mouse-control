'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
Since you will be using four models to build this project, you will need to replicate this file
for each of the models.

This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
import cv2
class Model_X:
    '''
    Class for a Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.network = None
        self.executable_network = None
        self.input_blob = None
        self.output_blob = None
        self.model_name = model_name
        self.device = device
        self.cpu_extensions = extensions

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        
        self.network = IENetwork(model = self.model_name+'.xml', weights = self.model_name +'.bin')
        ie = IEPlugin(device = self.device)
        
        if self.cpu_extensions:
            ie.add_cpu_extension(self.cpu_extensions)
       
        supported_layes = ie.get_supported_layers(self.network)
        model_layers  = self.network.layers
        self.input_blob  = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        try:
            self.check_plugin(supported_layes, model_layers)
        except Exception as e:
            print("Error in loading model ", e)
            exit()
        self.executable_network = ie.load(network=self.network, num_requests=1)
        return self.executable_network

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        
        Handled above.
        '''
        # Orig Width and Height
        height,width,_ = image.shape
        preprocessed_image = self.preprocess_input(image)

        return self.preprocess_output(self.executable_network.infer(inputs={self.input_blob:preprocessed_image}), width, height, image)

    def check_plugin(self, supported_layes, model_layers):
        '''
        TODO: You will need to complete this method as a part of the 
        standout suggestions

        This method checks whether the model(along with the plugin) is supported
        on the CPU device or not. If not, then this raises and Exception
        '''
        if len(supported_layes) < len(model_layers):
#             print("Some layers are not supported add CPU Extension")
#             print("Below layers are not supported = ")
#             print('\n'.join([layer for layer in model_layers.keys() if layer not in supported_layes.keys()]))
            raise Exception("Not supported layer. Please add CPU extensions and run the program again.")

    def preprocess_input(self, image):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        _,_, height, width = self.network.inputs[self.input_blob].shape
        image = cv2.resize(image, (width, height))
        image = image.transpose(2,0,1)
        image = image.reshape(1, 3, height, width)

        return image

    def preprocess_output(self, outputs,  width, height, image):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        output = outputs[self.output_blob]
#         print(output)
        output = output[0]
        print(output)
        eyes = []
        for i in range(0,8, 4):
            x_min = int(output[i] * width)
            y_min =  int(output[i+1] * height)
            x_max = int(output[i+2] * width)
            y_max =  int(output[i+3] * height)
           
            if x_min > x_max:
                t = x_min
                x_min = x_max
                x_max = t
                
            if y_min > y_max:
                t = y_min
                y_min = y_max
                y_max = t
           
            print((x_min, y_min), (x_max, y_max))
            eyes.append(image[y_min:y_max+10, x_min:x_max])
            frame = cv2.rectangle(image, (x_min, y_min), (x_max, y_max+10),(0, 0, 255), 1)

        return eyes[0],eyes[1], frame
    
'''
            To Test the code.
'''    
a = Model_X(model_name='/home/workspace/starter/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002',extensions='/home/workspace/starter/bin/libcpu_extension_sse4.so')

a.load_model()
image = cv2.imread("/home/workspace/starter/face_detected.png")
l,r,f = a.predict(image)
cv2.imwrite("left_eye.png", l)
cv2.imwrite("right_eye.png", r)
