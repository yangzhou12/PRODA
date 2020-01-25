# PRODA
Matlab source codes of the Probabilistic Rank-One Discriminant Analysis (PRODA) algorithm presented in the paper [Probabilistic Rank-One Discriminant Analysis via Collective and Individual Variation Modeling](https://ieeexplore.ieee.org/document/8481385).

## Usage
Face recognition with PRODA on 2D images from the FERET dataset:
```
Demo_PRODA.m
```

## Descriptions of the files in this repository  
 - DBpart.mat stores the indices for training (2 samples per class) /test data partition.
 - FERETC80A45.mat stores 320 faces (32x32) of 80 subjects (4 samples per class) from the FERET dataset.
 - Demo_PRODA.m provides example usage of PRODA for subspace learning and classification on 2D facial images.
 - PRODA.m implements the PRODA algorithm described in [paper](https://ieeexplore.ieee.org/document/8481385).
 - projPRODA.m projects 2D data into the subspace learned by PRODA.
 - sortProj.m sorts features by their Fisher scores in descending order.
 - logdet.m computes the logarithm of determinant of a matrix.

## Requirement
[Tensor toolbox v2.6](http://www.tensortoolbox.org/).

## Citation
If you find our codes helpful, please consider cite the following [paper](https://ieeexplore.ieee.org/document/8481385):
```
@article{
    zhou2019PRODA,
    title={Probabilistic Rank-One Discriminant Analysis via Collective and Individual Variation Modeling},
    author={Yang Zhou and Yiu-ming Cheung},
    journal={IEEE Transactions on Cybernetics},
    year={2020},
    volume={50},
    number={2},
    pages={627-639},
    doi={10.1109/TCYB.2018.2870440},
}
```
