clear; close all; clc

% 1st step = load dataset  2nd step = prepare the model   3rd step = train the model 4th step = evaluate the model
% using test data

img_width = 28;
img_height = 28;

n_filters = 9;
filter_size = 3;
num_labels = 10;

% Load Dataset
fprintf('Loading and Visualizing data');
load('ex4data1.mat');
m = size(X, 1);

sel = randperm(size(X,1));
sel = sel(1:200);

displayData(X(sel,:));
fprintf('Program Paused. Press enter to continue');
pause;

theta_conv_1 = initialize_theta(3,3);   % filter size as argument
theta_conv_2 = initialize_theta(3,3);   % filter size as argument
theta_conv_3 = initialize_theta(3,3);   % filter size as argument
theta_conv_4 = initialize_theta(3,3);   % filter size as argument
theta_conv_5 = initialize_theta(3,3);   % filter size as argument
theta_conv_6 = initialize_theta(3,3);   % filter size as argument
theta_conv_7 = initialize_theta(3,3);   % filter size as argument
theta_conv_8 = initialize_theta(3,3);   % filter size as argument
theta_conv_9 = initialize_theta(3,3);   % filter size as argument

% Prepare the model
for i = 1:m, 
  % Forward Propagation
  % RelU layer is included in the convolve function
  conv_1 = convolve(X(i,:), theta_conv_1);
  conv_2 = convolve(X(i,:), theta_conv_2);
  conv_3 = convolve(X(i,:), theta_conv_3);
  conv_4 = convolve(X(i,:), theta_conv_4);
  conv_5 = convolve(X(i,:), theta_conv_5);
  conv_6 = convolve(X(i,:), theta_conv_6);
  conv_7 = convolve(X(i,:), theta_conv_7);
  conv_8 = convolve(X(i,:), theta_conv_8);
  conv_9 = convolve(X(i,:), theta_conv_9);
  
  % Pooling Layer
  max_pool1 = pool(conv_1);
  max_pool2 = pool(conv_2);
  max_pool3 = pool(conv_3);
  max_pool4 = pool(conv_4);
  max_pool5 = pool(conv_5);
  max_pool6 = pool(conv_6);
  max_pool7 = pool(conv_7);
  max_pool8 = pool(conv_8);
  max_pool9 = pool(conv_9);
  