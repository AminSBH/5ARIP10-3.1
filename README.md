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

```python
#image paths
dir_train_x = 'data/training/images/'
label_path = 'data/training/1st_manual/'
dir_test_x = 'data/test/images/'
label_test_path = 'data/test/1st_manual/'
dir_valid_x = 'data/valid/images/'
label_valid_path = 'data/valid/1st_manual/'

images = list()
labels = list()
images_test = list()
labels_test = list()
images_valid = list()
labels_valid = list()

for file in Path(dir_train_x).iterdir():
    if not file.is_file():
        continue
    img_rgb = imageio.v2.imread(file)
    img_real = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    images.append(img_real)

for file in Path(label_path).iterdir():
    if not file.is_file():
        continue

    img_rgb = imageio.v2.imread(file)
    labels.append(img_rgb)
    
for file in Path(dir_test_x).iterdir():
    if not file.is_file():
        continue
    img_rgb = imageio.v2.imread(file)
    img_two = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    images_test.append(img_two)

for file in Path(label_test_path).iterdir():
    if not file.is_file():
        continue
    img_rgb = imageio.v2.imread(file)
    labels_test.append(img_rgb)
    
for file in Path(dir_valid_x).iterdir():
    if not file.is_file():
        continue
    img_rgb = imageio.v2.imread(file)
    img_two = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    images_valid.append(img_two)

for file in Path(label_valid_path).iterdir():
    if not file.is_file():
        continue
    img_rgb = imageio.v2.imread(file)
    labels_valid.append(img_rgb)

def resize(imgs,resize_height, resize_width):
    img_resize = []
    for file in imgs:
        img_resize.append(cv2.resize(file,(resize_height,resize_width)))
    return img_resize

resize_height, resize_width = (512, 512)
imgs = resize(images,resize_height, resize_width)
lbls = resize(labels,resize_height, resize_width)
imgs_test = resize(images_test,resize_height, resize_width)
lbls_test = resize(labels_test,resize_height, resize_width)
imgs_valid = resize(images_valid,resize_height, resize_width)
lbls_valid = resize(labels_valid,resize_height, resize_width)

X_train = np.array(imgs)
Y_train = np.array(lbls)
X_test = np.array(imgs_test)
Y_test = np.array(lbls_test)
X_valid = np.array(imgs_valid)
Y_valid = np.array(lbls_valid)

X_train = X_train.astype('float32')/255
Y_train = Y_train.astype('float32')/255
X_test = X_test.astype('float32')/255
Y_test = Y_test.astype('float32')/255
X_valid = X_valid.astype('float32')/255
Y_valid = Y_valid.astype('float32')/255

print("Pre-processing is finished.")
```

```python
X_train = np.transpose(X_train, (0, 3, 1, 2))
training_set = utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))

X_valid = np.transpose(X_valid, (0, 3, 1, 2))
valid_set = utils.data.TensorDataset(torch.Tensor(X_valid),  torch.Tensor(Y_valid))

X_test = np.transpose(X_test, (0, 3, 1, 2))
test_set = utils.data.TensorDataset(torch.Tensor(X_test),  torch.Tensor(Y_test))
```

```python
n_classes = 2
Model = smp.EfficientUnetPlusPlus(encoder_name='timm-efficientnet-b0', encoder_weights="imagenet", in_channels=3, classes=n_classes)
Model.to(device)
lr = 0.001
optimizer = optim.Adam(Model.parameters(), lr=lr)
criterion = tgm.losses.TverskyLoss(alpha=0.5, beta=5)
batchsize = 1
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=0.001, patience = 2)

data_loader_training = torch.utils.data.DataLoader(dataset=training_set, batch_size=batchsize, shuffle=True)
data_loader_valid = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batchsize, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=batchsize, shuffle=True)
```

```python
num_epochs = 20
#one example of the validation outputs
outputs = []
outputs_val = []
for epoch in range(num_epochs):
    #perform training loop
    Model.train()
    amount=0
    loss_value = 0
    avg_dice = 0
    loop = tqdm(data_loader_training)
    for (img,label) in loop:
        img, label = img.to(device), label.to(device)
        decoded = Model(img)
        loss = criterion(decoded.to("cpu"), label.type(torch.LongTensor).to("cpu"))
        loss_value += loss.item()
        optimizer.zero_grad()
        loss.backward()
        avg_dice += dice_score(input = decoded, target = label.type(torch.LongTensor).to("cpu").squeeze(1), use_weights=True)
        optimizer.step()
        amount+=1
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
    loss_value = loss_value/amount
    avg_dice = avg_dice/amount
    avg_dice = avg_dice.item()
    outputs.append((epoch, loss_value, avg_dice))
    print("Epoch", epoch, "Training loss:", loss_value, "Training dice:",avg_dice )
    #perform valid loop
    Model.eval()
    loss_value = 0
    amount = 0
    loop = tqdm(data_loader_valid)
    
    with torch.no_grad():
        avg_dice = 0
        for (img,label) in loop:
            img, label = img.to(device), label.to(device)
            decoded = Model(img)
            loss = criterion(decoded.to("cpu"), label.type(torch.LongTensor).to("cpu"))
            loss_value += loss.item()
            amount +=1
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            avg_dice += dice_score(input = decoded, target = label.type(torch.LongTensor).to("cpu").squeeze(1), use_weights=True)
            
    loss_value = loss_value/amount
    avg_dice = avg_dice/amount
    avg_dice = avg_dice.item()
    scheduler.step(loss_value)    
    outputs_val.append((epoch, loss_value, avg_dice))
    print("Epoch", epoch, "Validation loss:", loss_value, "Validation dice:",avg_dice ) 
```

```python
# Define the helper function
def decode_segmap(image, nc=2):
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=vessel
               (255, 255, 255)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb
```

```python
#show img+recon
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
 ```
 
 ```python
 Model.eval()
 outputs_test = []
 i= 0
 count = 0
 with torch.no_grad():
    avg_dice = 0
    for (img,label) in data_loader_test:
        img, label = img.to(device), label.to(device)
        decoded = Model(img)
        loss = criterion(decoded.to("cpu"), label.type(torch.LongTensor).to("cpu"))
        outputs_test.append((i, img, decoded,label.type(torch.LongTensor).to("cpu")))
        avg_dice += dice_score(input = decoded, target = label.type(torch.LongTensor).to("cpu").squeeze(1), use_weights=True)
        count+=1
        print("Image", i, "Loss:", loss.item()) 
        i+=1
avg_dice = avg_dice/count 
print(avg_dice.item())
```

```python
i = 5
orig = outputs_test[i][1].cpu().reshape(3,512,512).numpy().transpose(1,2,0)
label = outputs_test[i][3].cpu().reshape(512,512)
recon = outputs_test[i][2].cpu()
preds = torch.argmax(recon.squeeze(), dim=0).detach().cpu().numpy()
img = decode_segmap(preds)
i=2
orig3 = outputs_test[i][1].cpu().reshape(3,512,512).numpy().transpose(1,2,0)
recon3 = outputs_test[i][2].cpu()
preds3 = torch.argmax(recon3.squeeze(), dim=0).detach().cpu().numpy() 
label3 = outputs_test[i][3].cpu().reshape(512,512)
img3 = decode_segmap(preds3)

show([orig,img,label])
show([orig3,img3,label3])
```

```python
x = np.array(range(1,len(outputs)+1))
y = np.zeros(len(outputs))
y2 =np.zeros(len(outputs))

for i in range(0,len(outputs)):
    y[i] = outputs_val[i][2]
    y2[i] = outputs[i][2]
plt.title("Dice Score")
plt.plot(x, y, color="red", label="Validation Dice Score")
plt.plot(x, y2, 'g', label="Training Dice Score") 
plt.ylim([0.4, 1])
plt.xlabel("Epochs [-]")
plt.ylabel("Dice score [-]")
plt.legend(loc="lower left")

plt.show()

for i in range(0,len(outputs)):
    y[i] = outputs_val[i][1]
    y2[i] = outputs[i][1]

plt.title("Loss")
plt.plot(x, y, color="red", label="Validation Loss")
plt.plot(x, y2, 'g', label="Training Loss") 
plt.ylim([0, 0.3])
plt.legend(loc="lower left")

plt.xlabel("Epochs [-]")
plt.ylabel("Loss [-]")

plt.show()
```

## Stent Prediction
To view the main stent prediction model, see the code inside 'stent_size_prediction.ipynb'
```python

```

