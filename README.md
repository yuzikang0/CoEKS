# Combination-of-Experts with Knowledge Sharing for Cross-Task Vehicle Routing Problems (CoEKS)
## Paper
The paper associated with this codebase can be found here [Combination-of-Experts with Knowledge Sharing for Cross-Task Vehicle Routing Problems](https://openreview.net/forum?id=lHBs9mbgwp)



## 🚀 Installation

If you want to install all dependencies, please use `environment.yml` and `requirements.txt`.


## Running

The main training script can be called via:

```bash
python run.py
```

You can also specify the tasks used for training by configuring `subsample_problems` in `CoEKS/envs/mtvrp/generator.py`.



## Testing

You may use the provided test function:

```bash
python test.py --checkpoint checkpoints/CoEKS50/coeks.ckpt   
``` 

## Reference
If our work is helpful for your research, please cite our paper:
```
@inproceedings{yu2026combinationofexperts,
title={Combination-of-Experts with Knowledge Sharing for Cross-Task Vehicle Routing Problems},
author={Zikang Yu and Jinbiao Chen and Jiahai Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
}
```


### 🤗 Acknowledgements

- https://github.com/ai4co/routefinder
- https://github.com/RoyalSkye/Routing-MVMoE
- https://github.com/ai4co/rl4co
