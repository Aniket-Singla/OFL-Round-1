# One Fourth Labs Challenge

## Task 1: Digit / Letter Classifier

Labels:
0 -> Digits
1 -> Letters 

### Importing Model for task 1: <br />

Pickle requires class to be defined before loading a model. <br />
This model excepts batch of shape [<batch_size> , 1, 28, 28]

```
import torch
from torch import nn

class LeNet_T1(nn.Module):
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


model = torch.load('task1_model.pth',map_location='cpu')
```
Accuracy Achieved =  91.35 %
## Task 2: Vowel/Consonant and Even/Odd Classifier

I have split the both train and test datasets into Letters and Digits for training on two different models.

For Training Labels used : <br />
Labels Digits:
0 -> Even
1 -> Odd

Labels Letters: 
0 -> Vowels
1 -> Consonants

For Testing (Evaluation) Labels Used: <br />
0 -> digit, even
1 -> digit, odd
2 -> letter, vowel
3 -> letter, consonant

### Importing model for Task 2:
This model excepts inputs strictly of shape [1,1,28,28]. This is not a nn.Module model but a mixture of 3 models to predict the final results as given in problem statement.
```
import torch
from torch import nn

class LeNet_T1(nn.Module):
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

class LeNet_T2(nn.Module):
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


model_digits = torch.load('task2_digits_model.pth',map_location='cpu')
model_letters = torch.load('task2_letters_model.pth',map_location='cpu')
model_t1 = torch.load('task1_model.pth',map_location='cpu')

def model(image):
  # run digits/ letter classifier
  if image.shape[0]==torch.Size([1,1,28,28]):
      raise Exception('This Model support only images of shape [1,1,28,28]')
  output_1 = model_t1(image)
  _, pred_1 = torch.max(output_1,1)
  if pred_1==0:
    # run even/odd classifier
    output_2 = model_digits(image)
    _, pred_2 = torch.max(output_2,1)
    return 0 if pred_2==0 else 1
  else:
    # run vowel/ consonant classifier
    output_2 = model_letters(image)
    _, pred_2 = torch.max(output_2,1)
    return 2 if pred_2==0 else 3
```

Accuracy Achieved =  87.65%

## Task 3: Character Classifier 

### Labels : 
0 -> 0 <br />
1 -> 1 <br />
2 -> 2 <br/>
3 -> 3 <br/>
4 -> 4 <br/>
5 -> 5 <br/>
6 -> 6 <br/>
7 -> 7 <br/>
8 -> 8 <br/>
9 -> 9 <br/>
10 -> A <br/>
11 -> B <br/>
12 -> C <br/>
13 -> D <br/>
14 -> E <br/>
15 -> F <br/>
16 -> G <br/>
17 -> H <br/>
18 -> I <br/>
19 -> J <br/>
20 -> K <br/>
21 -> L <br/>
22 -> M <br/>
23 -> N <br/>
24 -> O <br/>
25 -> P <br/>
26 -> Q <br/>
27 -> R <br/>
28 -> S <br/>
29 -> T <br/>
30 -> U <br/>
31 -> V <br/>
32 -> W <br/>
33 -> X <br/>
34 -> Y <br/>
35 -> Z <br/>
36 -> a <br/>
37 -> b <br/>
38 -> d <br/>
39 -> e <br/>
40 -> f <br/>
41 -> g <br/>
42 -> h <br/>
43 -> n <br/>
44 -> q <br/>
45 -> r <br/>
46 -> t <br/>

### Importimg Model: <br />
This Model expects batch images of size [<batch_size>, 1,28, 28]. 
```
import torch
from torch import nn

class LeNet(nn.Module):
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


model = torch.load('task3_model.pth',map_location='cpu')

```
Accuracy Achieved =  81.63%
