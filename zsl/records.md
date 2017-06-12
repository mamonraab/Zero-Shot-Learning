#Failed Model
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
