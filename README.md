# Group 3.1 Stent size prediction model
In this repository both the segmentation model and the stent size prediction model are given. For the segmentation model, the EfficientUnet++ architecture is used. For the stent size prediction, the main component was the Autogluon package. How to run the scripts in order to achieve the same results will be explained here.
## Installation
In order to install the github repository, run the following code inside your Visual Studio Code Terminal
```python
git clone https://github.com/AminSBH/5ARIP10-3.1.git
```
The following packages need to be installed:


## Vessel Segmentation

To view the main vessel segmention model, see the code inside 'EfficiientNetNoteBook.ipynb'
To facilitate the training of the model, the gpu is used, as signified by the following lines
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)
```
To fit the Efficient Unet ++
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)
```
## Stent Prediction
To view the main stent prediction model, see the code inside 'stent_size_prediction.ipynb'
```python

```

