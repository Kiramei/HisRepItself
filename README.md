## This is for Kiramei's model Learning
## History Repeats Itself: Human Motion Prediction via Motion Attention
This is the code for the paper

Original Author Wei Mao, Miaomiao Liu, Mathieu Salzmann. 
[_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://arxiv.org/abs/2007.11755). In ECCV 20.

Wei Mao, Miaomiao Liu, Mathieu Salzmann, Hongdong Li.
[_Multi-level Motion Attention for Human Motion Prediction_](https://arxiv.org/abs/2106.09300). In IJCV 21.

### Dependencies
To create the environment:
```shell
conda create -n HRI
conda activate HRI
```

To install python and pytorch:
```shell script
conda install python==3.6.13
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.1 cudatoolkit=11.3.1 -c pytorch -c conda-forge
```

To install dependencies:
```shell script
python -m pip install -r requirements.txt 
```

### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[AMASS](https://amass.is.tue.mpg.de/en) from their official website..

Directory structure:
```shell script
amass
|-- ACCAD
|-- BioMotionLab_NTroje
|-- CMU
|-- ...
`-- Transitions_mocap
```
[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
Put the all downloaded datasets in ./datasets directory.

### Training
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train,

#### For 3d points Model,

Short Term Model
```bash
python main_h36m_3d.py  --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66
```
Long Term Model
```bash
python main_h36m_3d.py  --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66
```
#### For angle Model,

Short Term Model
```bash
python main_h36m_ang.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 48
```
Long Term Model
```bash
python main_h36m_ang.py --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 48
```

### Evaluation
To evaluate the pretrained model,
#### For 3d points Model,
Short Term Model
```bash
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --ckpt ./checkpoint/main_h36m_3d_in50_out10_ks10_dctn20
```
Long Term Model
```bash
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --ckpt ./checkpoint/main_h36m_3d_in50_out25_ks10_dctn35
```
#### For angle Model,
Short Term Model
```bash
python main_h36m_ang_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --ckpt ./checkpoint/main_h36m_ang_in50_out10_ks10_dctn20
```
Long Term Model
```bash
python main_h36m_ang_eval.py --is_eval --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --ckpt ./checkpoint/main_h36m_ang_in50_out25_ks10_dctn35
```


### Licence
MIT
