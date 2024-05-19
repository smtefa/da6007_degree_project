# da6007_degree_project
Code for the bachelor's degree project in computer science.

* Requirements:
  * Python 3.11.7 or newer.
  * PyTorch library.
  * [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

* The code is built from several files:
  * File [code/resnet18k.py](code/resnet18k.py) is responsible for the creation and trainig of the
    ResNet-18 model, as well as the plotting of the output geometry.
  * File [code/data.py](code/data.py) handles the CIFAR-10 data set; including its downloading, conversion
    to data loaders and introduction of noisy labels.
  * File [code/helper.py](code/helper.py) includes some technical helper functions, mainly
    for plotting.
    
* How to recreate the results:
  * In [code/main.py](code/main.py), adjust the parameters as desired then run the code.

