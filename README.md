# Meta-Deformation

Here is the [Training Data](https://drive.google.com/file/d/1HVReM43YtJqhGfbmE58dc1-edI_oz9YG/view), [Testing Data](https://drive.google.com/file/d/1mXE0mShpMUorQUfM3uZuJ6Si_pIl54ED/view?usp=sharing), [Template](https://drive.google.com/drive/folders/1meehKGn0pq4pm9ye_Qn-BzCfno0UpBlt?usp=sharing), [Inference Data](https://drive.google.com/drive/folders/1ThR25XKE1X2kApWD0oL61n7c90xMXCEc?usp=sharing)

Deploy the environment

``` shell
mkdir data
cd data
cp path/to/datas_surreal_train.pth .
cp path/to/datas_surreal_test.pth .
cp path/to/template_set_1 .
cp path/to/MPI-FAUST . 
```

To train this model

```shell
python training/train.py
```

To do inference on MPI-FAUST

```shell
python inference/script.py
```

