import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

#importing model
interpreter = tflite.Interpreter(model_path='')

#opencv reads image 
path1 = ''
img = cv2.imread(path1)[:,:,::-1]
img = cv2.resize(img,(150,150))
test=np.expand_dims(img,axis=0)
test_image = test.astype(np.float32)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = test_image
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

if ((output_data[0][0]) == 1.0):
    print ('step_1')
elif ((output_data[0][1]) == 1.0):
    print ('step_2')
elif ((output_data[0][2]) == 1.0):
    print ('step_3')
elif ((output_data[0][3]) == 1.0):
    print ('step_4')
elif ((output_data[0][4]) == 1.0):
    print ('step_5')
else:
    print ('no_hands')