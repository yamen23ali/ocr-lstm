# pml
## Setup Local Environment 

1- Get the repo

    git clone git@github.com:safaa-alnabulsi/pml.git
    cd pml

2- Create virtual enviroment 

    conda create --name pmlpy3.7 python=3.7
    conda activate pmlpy3.7
    pip install -r requirements.txt
    jupyter notebook

## Get the data
1- Create a folder for the dataset and download it:

    mkdir dataset
    cd dataset
    wget https://zenodo.org/record/1344132/files/GT4HistOCR.tar
    
2- Unpack the data:   

    tar -xvf 'GT4HistOCR.tar'
    for f in corpus/*.tar.bz2; do echo $f && tar xjf $f; done

3- Clean tar files:

    rm -r GT4HistOCR.tar corpus

## Useful Commands

```

qsub -V
qsub -V -cwd
qsub -V -cwd -j
qsub -V -cwd -j y -o dirname
qsub -V -cwd -j y -o dirname -l cuda=1 #to add more resources
qsub -V -cwd -j y -o dirname -l cuda=1 run.sh params # to add a script to the queue
qstat -u "*" #look at queue of all users
qstat -u “id” #look at jobs of a specific user
setfacl
setfacl -R -m dirname:rwX filename # to change permission of the folder
getfacl # to check the permission of a folder

```

## References:

- [High-Performance OCR for Printed English and Fraktur using LSTM Networks](https://ieeexplore.ieee.org/document/6628705)
- [Ground Truth for training OCR engines on historical documents in German Fraktur and Early Modern Latin](https://arxiv.org/pdf/1809.05501.pdf)
   - https://github.com/tmbdev/ocropy/wiki
   - https://www.cis.uni-muenchen.de/ocrworkshop/program.html
   - http://cistern.cis.lmu.de/ocrocis/tutorial.pdf
   - https://github.com/tesseract-ocr/tesseract/wiki/Training-Tesseract
- [warp-ctc PyTorch code](https://github.com/baidu-research/warp-ctc)
