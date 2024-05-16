# da6007_degree_project
Code for the bachelor's degree project in computer science.

* Requirements:
  * Python 3
  * [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

* The code is built from several files:
  * File [code/resnet18k.py](code/resnet18k.py) is responsible for the creation and trainig of the
    ResNet-18 model, and it can also plot the output geometry.
  * File [code/data.py](code/data.py) handles the CIFAR-10 data set; including its downloading, conversion
    to data loaders and also introduction of noisy labels.
  * File [code/helper.py](code/helper.py) includes some technical helper functions for other files, mainly
    for plotting.
    
* How to recreate the results:
  * In [code/main.py](code/main.py), adjust the parameters as desired then run the code.

* To check the results obtained in the paper, see
  [experiments/](experiments).
