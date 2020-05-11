# Evolving Neural Network through a Reverse Encoding Tree

News: Our Paper has been accepted to IEEE CEC 2020. An updated version could be found out [here](https://arxiv.org/abs/2002.00539). Feel free to contact us for experiement details.


<img src="https://github.com/HaolingZHANG/ReverseEncodingTree/blob/master/Figure/demo_RET2020.png" width="300">

Code for Python 3.7 implementation (in the PyCharm) of **Reverse Encoding Tree** from the [paper](https://arxiv.org/abs/2002.00539).
## Getting Started
The library is divided into two parts.
In the **benckmark** part, you will easy easily understand the principle of our strategy and its difference from other strategies. 
In the **evolution** part, you can use it for many tasks of NeuroEvolution.

We have further integrated **neat-python** in **evolution/bean**.
The files in the **example** folder describe how to use the original NEAT to finish the well-accepted tasks.
**tasks** folder includes all the execution documents in the experiments mentioned in the paper.

### Prerequisites
- [neat-python](https://pypi.org/project/neat-python/) -- version 0.92
- [gym](https://pypi.org/project/gym/) -- version 0.14.0
- [box2d](https://pypi.org/project/Box2D/) -- version 2.3.2
- [matplotlib](https://pypi.org/project/matplotlib/) -- version 3.1.1
- [pandas](https://pypi.org/project/pandas/) -- version 0.25.1
- [numpy](https://pypi.org/project/numpy/) -- version 1.17.1

### Building a Bi-NEAT
We have 6 additional hyper-parameters in the configure.
- **max_node_num** in the **network parameters**: maximum numnber of node in all the generated neural networks, it describes the range of phenotypic landscape.
- **init_distance** in the **Reproduction**: initial distance describes the minimum distance between each of the two neural networks in the initial (first) generation.
- **min_distance** in the **Reproduction**: minimum distance describes the minimum distance between each of the two neural networks after the initial (first) generation.
- **correlation_rate** in the **Reproduction**: correlation rate describes the demarcation line between positive and negative correlation coefficient. The default value is **-0.5**. If the correlation coefficient less than correlation rate, it is positive.
- **search_count** in the **Reproduction**: search count describes the maximum number of searches required when adding a novel neural network.
- **cluster_method** in the **Reproduction**:  Alternative clustering methods, the default is **kmeans++**. We have **kmeans**, **kmeans++**, **birch** and **spectral** options.

You need to create a configure before running, the document including original settings is shown in [https://readthedocs.org/projects/neat-python/](https://readthedocs.org/projects/neat-python/).

After creating the configure:
```python
config = neat.Config(genome.GlobalGenome, bi.Reproduction, species_set.StrongSpeciesSet, neat.DefaultStagnation, "your configure path")
neat.Population(config)
```
### Continous Learning Environments

- Cartport-v0

<img src="https://github.com/HaolingZHANG/ReverseEncodingTree/blob/master/Figure/cartpole.gif" width="300">

- LunarLander-v2

<img src="https://github.com/HaolingZHANG/ReverseEncodingTree/blob/master/Figure/lunar_lander_success_example.gif" width="300">

If you think this repo helps or being used in your research, please consider refer this paper. Thank you.

- [Evolving Neural Networks through a Reverse Encoding Tree](https://arxiv.org/abs/2002.00539), Arxiv 2002.00539

````
@article{zhang2020evolving,
  title={Evolving Neural Networks through a Reverse Encoding Tree},
  author={Zhang, Haoling and Yang, Chao-Han Huck and Zenil, Hector and Kiani, Narsis A and Shen, Yue and Tegner, Jesper N},
  journal={arXiv preprint arXiv:2002.00539},
  year={2020}
}
````


Haoling Zhang, Chao-Han Huck Yang, Hector Zenil, Narsis A. Kiani, Yue Shen, Jesper N. Tegner
# Contributors
[Haoling Zhang](https://github.com/HaolingZHANG), [Chao-Han Huck Yang](https://github.com/huckiyang)
