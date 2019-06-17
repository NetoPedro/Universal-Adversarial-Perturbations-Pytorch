# Universal-Adversarial-Perturbations-Pytorch

Implementation of (https://arxiv.org/abs/1610.08401) for the CS-E4070 - Special Course in Machine Learning and Data Science: Advanced Topics in Deep Learning course at Aalto University, Finland. 



## Paper interpretation 

Adversarial examples are a big issue of machine learning models, especially neural networks. In this paper, the authors try to exploit some characteristics of this problem and the properties of the adversarial examples. 

The paper has shown some impressive results: 

- It has shown that the state of the art neural networks can be fooled with a high probability on most of the images with the same computed adversarial example.

- The paper also proposed the pseudo code of the algorithm implemented in this repository to find the perturbations that can fool with high probability most of the images shown to a neural network. 

- It is also shown and possible to see in this implementation that a rather small number of training samples is enough to generate a perturbation capable of fooling the network with high probability on the remaining data.  

- Although this part is not compared with the code in this repository, it is shown that these perturbations are somewhat capable to generalize also to other architectures. This is what the authors call "Doubly universal", with respect to the data and to the architecture of the network. 

- Finally, it is shown that there is a geometric correlation behind these examples that make neural networks highly vulnerable if anyone with malicious intent tries to affect the system. 


Other interesting topics on this paper: 

- This implementation doesn't try to find a data point specific perturbation as previous algorithms do.  Instead, it tries to find a perturbation that generalizes well to most data points. 

- The perturbations are based on some data distribution and not directly intended to be generalized across different models.


### Universal Perturbations

The main goal is to find a perturbation where: 

- K'(x+v)  != k'(x) for most x ~ distribution

The above formula can be translated to a perturbation v that when added to x, changes the output of K' for most of the data points x drawn from the distribution. 

The generated perturbation 'v' is subject to some constraints: 

- ||v||p <= E 

- Fooling rate >= 1 - delta

The first constraint is responsible to manage and regulate how small is the vector v. The second one ensures that the fooling rate for all images is above the defined threshold. On this implementation, the fooling rate is calculated using all the images on the test set.  

### Algorithm



### Vulnerability to Universal Perturbations 

The project is runnable from the main file. 



