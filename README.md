# MRI-Reconstruction-Algorithms

This repository contains implementations of several common MRI reconstruction algorithms. The goal is to reproduce these algorithms and make the code available to help researchers and students.

## Algorithms

- Filtered Back Projection (FBP)
- Linear Reconstruction
   - Parallel Reconstruction
      - Controlled Array
        - SMASH
        - SENSE
        - GRAPPA
- Iterative Reconstruction
   - Least Squares 
   - Sparse MRI（Compressed Sensing——CS、Dictionary Learning）
   - Low Rank
- AI Reconstruction
   - MoDL
   - ISTA-NET

## Data Prepare

Download data from the link fastMRI: https://fastmri.org/dataset/

MoDL：Link---https://pan.baidu.com/s/1LIf_3KQEuOVG7JjVYhhRXQ?pwd=1516  Password---1516
ISTA-NET: Link---https://pan.baidu.com/s/13RhJJmoK17M5vP3_r7HDrA?pwd=1516  Password---1516

## Code Source

MoDL(Model Based Deep Learning Architecture for Inverse Problems): https://github.com/hkaggarwal/modl
ISTA-Net(Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing): (Pytorch) https://github.com/jianzhangcs/ISTA-Net-PyTorch
                                                                                          (Tensorflow) https://github.com/jianzhangcs/ISTA-Net

## Usage

Each algorithm is implemented as a Jupyter Notebook with detailed comments. The notebooks can be run locally or viewed on GitHub. 

Data and sample results are included to demonstrate the algorithms. Code is well documented and tested.

## Contributing

New algorithm implementations are welcome. Create an issue first to discuss any proposed changes or additions.

## Acknowledgments

Thank you to original authors of the algorithms. Code builds upon work from public MRI reconstruction packages.

Let me know if you have any other questions!

