##Semantic Space - Word2Vec
####Failed Model
[learning rate=1e-3
iter = 400
activation = sigmoid
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.0230911709368
training time = 2.8988014857 mins]

[learning rate=1e-3
iter = 40000
activation = sigmoid
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.00209919735789
training time = 47.4634619633 mins
loss plot=img2.png]

conclusion - optimal no. of iterations required ~ 3000

[learning rate=1e-3
iter = 4000
activation = no activation
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.0064210742712
training time = 3.13521726529 mins
loss plot=img3.png]

[learning rate=1e-3
iter = 6000
activation = no activation
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.00127598270774
training time = 4.01444385052 mins
loss plot=img4.png]

#Running Model
[*Training*
learning rate=1e-3
iter = 2000
activation = no activation(linear output)
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.798682868481
training time = 3.3570369482 mins
loss plot=None
*Testing*
Final Testing Accuracy=0.34142395854
testing time = 2.093509535 mins
]

[*Training*
sample per class is same
learning rate=1e-3
iter = 2000
activation = no activation(linear output)
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.792179465294
training time = 2.96647116741 mins
loss plot=img6.png
*Testing*
Final Testing Accuracy= 29.0291249752 %
testing time = 1.9132316192 mins
]

[*Training*
sample per class is same
learning rate=1e-3
iter = 3000
activation = no activation(linear output)
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.833916425705
training time = 3.62446976503 mins
loss plot=img5.png
*Testing*
Final Testing Accuracy=38.1877034903 %
testing time = 2.80982198318 mins
]

###Semantic Space- Attribute Space
[*Training*
learning rate=1e-3
iter = 6000
activation = no activation(linear output)
n_layers = 2
optimizer = Adam
Final Training Accuracy=0.802263855934
training time = 3.20129233599 mins
loss plot=None
*Testing*
Final Testing Accuracy= 8.81877020001 %  (may be the reason can be overfitting of the data)
testing time = 2.73782943487 mins
]

[*Training*
sample per class is same|| Attributes are scaled so that their mean is zero
learning rate=1e-3
iter = 3000
activation = no activation(linear output)
n_layers = 2
optimizer = Adam
Final Training Accuracy=83.7044656277 %
training time = 3.03020784855 mins
loss plot=None
*Testing*
Final Testing Accuracy= 11.1488670111 %  (may be the reason can be overfitting of the data)
testing time = 2.71812096437 mins
]