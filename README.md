# SGMC: Spectral Graph Matrix Completion

Code for AAAI21 paper "Scalable and Explainable 1-Bit Matrix Completion via Graph Signal Learning". 


## Data Format

The implementation is desiged for top-N recommendations on implicit data, and thus it takes user-item pairs as input:

```
uid,sid
1,1
```

## Installation

The program requires Python 3.7+ with NumPy, SciPy, Pandas and PySpark.

**Note:**  
- To achieve the best performance, we highly recommend to use 
[NumPy/SciPy with MKL Intel](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-mkl-and-third-party-applications-how-to-use-them-together.html)


## Train and Test

After specifying the location of files <ins>*train.csv/test_tr.csv/test_te.csv*</ins> in runme.sh, it is quite simple to train and evaluate the model by
```
bash runme.sh
```


## Citation

If you find our code useful for your research, please consider cite.

```
@inproceedings{chen2021scalable,
  title={Scalable and Explainable 1-Bit Matrix Completion via Graph Signal Learning},
  author={Chen, Chao and Li, Dongsheng and Yan, Junchi and Huang, Hanchi and Yang, Xiaokang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI '21)},
  volume={35},
  number={8},
  pages={7011--7019},
  year={2021}
}
```
