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
        print("Gaze estimation model loaded")
        return self.executable_network

    def predict(self,left_eye , right_eye, head_pose):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        
        Handled above.
        '''
        # Orig Width and Height
        input_gaze = {
            'left_eye_image':self.preprocess_input(left_eye,'left_eye_image' ),
            'right_eye_image':self.preprocess_input(right_eye,  'right_eye_image'),
            'head_pose_angles':head_pose
        }
        return self.preprocess_output(self.executable_network.infer(inputs=input_gaze))

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

    def preprocess_input(self, image, input_layer_name):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        _,_, height, width = self.network.inputs[input_layer_name].shape
        image = cv2.resize(image, (width, height))
        image = image.transpose(2,0,1)
        image = image.reshape(1, 3, height, width)

        return image

    def preprocess_output(self, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        output = outputs[self.output_blob]
        return output[0]
    
'''
            To Test the code.
'''    
# a = Model_X(model_name='/home/workspace/starter/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002',extensions='/home/workspace/starter/bin/libcpu_extension_sse4.so')

# a.load_model()
# image = cv2.imread("/home/workspace/starter/bin/face.png")
# l_eye  = cv2.imread("/home/workspace/starter/left_eye.png")
# r_eye  = cv2.imread("/home/workspace/starter/right_eye.png")
# a.predict(l_eye, r_eye, [-12.527571, 36.17366, 6.397007])
# cv2.imwrite("face_detected.png", face)
# cv2.imwrite("face_box.png", frame)
