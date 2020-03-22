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
        boxes = output[0][0]
        face_count = 0
        face_coord_list = []
        for box in boxes:
            conf = box[2]
            # Person class filter
            if conf > 0.5:

                x_min = int(box[3] * width)
                y_min =  int(box[4] * height)
                x_max = int(box[5] * width)
                y_max =  int(box[6] * height)
                face_count+=1
                face_coord_list.append([x_min, y_min, x_max, y_max])
                frame = cv2.rectangle(image, (x_min, y_min), (x_max, y_max),(0, 0, 255), 1)


#         print("Drawing box done!")
        print("Total number of faces detected ", face_count)
        print("Selecting first face to track mouse pointer.")
        if face_count != 0:
            cropped_face = image[y_min:y_max, x_min:x_max]
        return cropped_face, frame
    
    
'''
            To Test the code.
'''    
# a = Model_X(model_name='/home/workspace/starter/intel/face-detection-adas-binary-0001/model/face-detection-adas-binary-0001',extensions='/home/workspace/starter/bin/libcpu_extension_sse4.so')

# a.load_model()
# image = cv2.imread("/home/workspace/starter/bin/face.png")
# face, frame = a.predict(image)
# cv2.imwrite("face_detected.png", face)
# cv2.imwrite("face_box.png", frame)
