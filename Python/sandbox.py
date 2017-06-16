from __future__ import absolute_import,division,print_function

from Mesh import Mesh
from PIL import Image
import numpy as np
from helpers.visualization import convert_labelled_output

# test = Mesh('R:/Projects/Nashville/Nash/Productions/Etienne_simpler/Data/Model/Model.obj')
# test.labelise('C:/Users/Etienne.Houze/Documents/Source/3DLearning/C++/Result/Projections.txt','C:/Users/Etienne.Houze/Documents/Source/3DLearning/C++/Result/labels')
# test.save_to_ptx('mesh.ptx','labels.txt')

# im = Image.open(r"C:\Users\Etienne.Houze\Documents\Source\3DLearning\C++\Result\Depth\IMG_4720.png")
# im_array = np.asarray(im)
# print("ha")

convert_labelled_output("./scripts/model_test/watch", num_labs=3)