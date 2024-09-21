# Welcome to our toolbox for cross-modal retrieval！ 👋👋👋

Our platform provides implementations of representative cross-modal retrieval methods, 
offering a comprehensive resource for researchers to experiment with and compare different approaches. 
By presenting these tools in an accessible format, we aim to streamline the process of experimental validation and performance evaluation. 

# Introduction
In this toolbox, we have integrated some representative cross-modal retrieval methods. Since these methods utilize different programming languages (such as Matlab and Python), development frameworks (such as TensorFlow and Torch), and diverse dependencies, we have organized them into three main environments:  
1. **Matlab**, to support all shallow cross-modal retrieval methods;  
2. **TensorFlow**, to support some deep cross-modal retrieval methods;  
3. **Torch**, to support some deep cross-modal retrieval methods.

Once the toolbox and datasets are stored and the above environments is set up, you can invoke the integrated cross-modal retrieval methods by running the `train.sh` file within the toolbox.


# Instruction
Below, we showcase how to use the cross-modal retrieval toolbox we developed.

## Install the toolbox
This guide provides step-by-step instructions for installing the cross-modal retrieval toolbox into the `./` directory on your local machine.

1. Clone the repository  
```bash
git clone https://github.com/cross-modal-retrieval/cross-modal-retrieval.git
```

2. Navigate to the toolbox directory  
```bash
cd cross-modal-retrieval/Toolbox
```

## Configure datasets
The dataset for the Cross-Modal Retrieval Toolbox is available via Baidu Yun. Follow the steps below to download and install the dataset into the `cross-modal-retrieval/Toolbox/data` directory.

1. Create the directory
```bash
mkdir -p ./data
```

2. Download datasets
- [Baidu Yun Link](https://pan.baidu.com/s/1QnC4ZyvjKOakKtUR9Cqd4A)
- **Code:** `6fb3`
Once the datasets are downloaded, move the downloaded dataset files to the `./data` directory.

3. Download pre-trained models
- [Baidu Yun Link](https://pan.baidu.com/s/1fJm8t9-YoSJ-4wSRv7ugxA)
- **Code:** `6fb3`
Once the models are downloaded, move the downloaded model files to the `./data` directory.

4. Verify the datasets and models
```bash
ls ./data
```

## Configure environments
This guide provides step-by-step instructions for setting up the environment for the cross-modal retrieval toolbox.  The toolbox supports different environments for Matlab, TensorFlow, and Torch, and you can configure them accordingly.

1. Install the TensorFlow environment
``` bash
conda create -n cmr_toolbox_tf python=3.7.16
conda activate cmr_toolbox_tf
bash environments/install_env_tf.sh
```

2. Install the Torch environment
``` bash
conda create -n cmr_toolbox_torch python=3.7.16
conda activate cmr_toolbox_torch
bash environments/install_env_torch.sh
```

3. Install the Matlab environment


## Use the toolbox


# Integrated methods
## List of integrated methods

## Note
Our toolbox is still in its initial version, and more cross-modal retrieval methods are continuously being added. If you have any questions, please feel free to leave us a message!

# Citation
If you find this toolbox helpful, please cite our paper:
```
@article{li2023cross,
      title={Cross-modal retrieval: a systematic review of methods and future directions},
      author={Li, Fengling and Zhu, Lei and Wang, Tianshi and Li, Jingjing and Zhang, Zheng and Shen, Heng Tao},
      journal={arXiv preprint arXiv:2308.14263},
      year={2023}
      }
```
