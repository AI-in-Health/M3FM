# System Requirements
## Hardware requirements
M3FM package requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 2 GB of RAM. For optimal performance, we recommend a computer with the following specs:
```
RAM: 16+ GB
CPU: 4+ cores, 3.3+ GHz/core
GPU: 2+ V100
```
The runtimes below are generated using a computer with the recommended specs.

## Software requirements
This package is supported for Linux. The package has been tested on the following system:
```
Linux: Ubuntu 16.04
```
## Environment
Clone the repo
```
git clone https://github.com/ai-in-health/M3FM
```
Install dependencies
```
conda create -n M3FM python==3.9
conda activate M3FM
# install a proper version of PyTorch
# see https://pytorch.org/get-started/previous-versions/
pip install pytorch>=1.10.1 torchvision>=0.11.2 torchaudio>=0.10.1 pytorch-cuda==11.8 -c pytorch -c nvidia

# install the rest dependencies
pip install -r requirements.txt
```
which should install in about 5 mins. We can run the code under `torch==2.2.1`. Other versions may work.

## Data
```
The root of this repo/
    data
    ├── mimic_cxr
    │   ├── annotation.json             # MIMIC_CXR dataset
    │   ├── images
    │   │   ├── p10               
    │   │   ├──  .
    │   │   ├──	 .
    │   │	└── p19
    ├── finetune
    │   ├── annotation.json             # Downstream dataset
    │   ├── images
    │   │   ├── patient 1               
    │   │   ├──  .
    │   │   ├──	 .
    pretrained
    ├── resnet
    │		└── pytorch_model.bin
    │
    decoder_config
    │		└── decoder_config.pkl      # configuration file for model
    │
    checkpoint	
    │		└── model_best.pth
    │		└── last_checkpoint.pth
    │
    └── ...
```
The `data` folder contains the following subfolders:
- The `mimic_cxr` folder contains MIMIC-CXR image data and corresponding annotation (eg., image id, report etc...).
- The `finetune` folder contains the image data and corresponding annotation (eg., image id, report etc...) of the downstream dataset. 
- The `pretrained` folder contains the initialized weights for our model which will download automatically when you run the code.
- The `decoder_config` and `checkpoint` folders contain the configuration file of the model, and its weights, respectively.

  
## Training
Here is an example of running command:
```
python M3FM.py
```

## Inference

Here is an example of running command:

Generate reports in English:
```
python inference.py --language English
```
Generate reports in Chinese
```
python inference.py --language Chinese
```
Generate reports in English and Chinese
```
python inference.py --language All
```

## Notes
1. To evaluate report generation, ensure that your system has installed JAVA. Here is an example:
   - Download from the official website (https://www.java.com/en/download/manual.jsp) to obtain, e.g., jdk-8u333-linux-x64.tar.gz
   - Unzip the file by running `tar -zxvf jdk-8u333-linux-x64.tar.gz`, and you will see the jre folder
   - Write the following lines to ~/.bashrc:
     - `echo "export JRE_HOME=path/to/jre" >> ~/.bashrc`
     - `echo "export PATH=${JRE_HOME}/bin:$PATH" >> ~/.bashrc`
   - Activate the settings by running `source ~/.bashrc`
   - See if the java has been installed: `java -version`
2. You should install packages `pycocoevalcap` and `pycocotools` (included in `requirement.txt`).
3. When calculating the `SPICE` metric, the code will try to automatically download two files `stanford-corenlp-3.6.0.jar` and `stanford-corenlp-3.6.0-models.jar`, and save them to ${pycocoevalcapPath}/spice/lib/. If you encounter a network issue, you can prepare these two files by yourself:
   - Download a zip file from https://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
   - Unzip it to get the above two files
   - Run `pip show pycocoevalcap` to see where the package has been installed
   - Move the two files to ${pycocoevalcapPath}/spice/lib/
4. To evaluate report generation, you should install the `stanfordcorenlp` package (included in `requirement.txt`), and download [stanford-corenlp-4.5.2](https://stanfordnlp.github.io/CoreNLP/history.html). The following is an example. Note that we set `corenlp_root = data/stanford-corenlp-4.5.2` in `configs/__init__.py`.
```
wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.2.zip --no-check-certificate
wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.2-models-german.jar --no-check-certificate
wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.2-models-french.jar --no-check-certificate

unzip stanford-corenlp-4.5.2.zip -d data/
mv stanford-corenlp-4.5.2-models-german.jar data/stanford-corenlp-4.5.2/
mv stanford-corenlp-4.5.2-models-french.jar data/stanford-corenlp-4.5.2/
rm stanford-corenlp-4.5.2.zip
```
