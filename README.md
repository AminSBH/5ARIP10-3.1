# Group 3.1 Stent size prediction model
## Installation
In order to install the github repository, run the following code inside your Visual Studio Code Terminal
```python
git clone https://github.com/AminSBH/5ARIP10-3.1.git
```

## Vessel Segmentation

To view the main vessel segmention model, see the code inside 'EfficiientNetNoteBook.ipynb'
To facilitate the training of the model, the gpu is used as signified by the following lines
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)
```

## Stent Prediction
To view the main stent prediction model, see the code inside 'stent_size_prediction.ipynb'
```python

```
