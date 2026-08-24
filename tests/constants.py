import numpy as np

IMAGE_INPUT_SHAPE=(38, 96, 96, 3) # Example shape of an RGB image
test_image = np.random.rand(*IMAGE_INPUT_SHAPE[1:]).astype(dtype=np.uint8) # example RGB image 
LIP_FEATURE_INPUT_SHAPE = (38, 20, 2)
test_lip_features = np.random.rand(*LIP_FEATURE_INPUT_SHAPE)
