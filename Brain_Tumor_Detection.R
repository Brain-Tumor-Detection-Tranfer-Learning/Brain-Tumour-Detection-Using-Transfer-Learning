base_dir <- list.dirs(path = "D:/vit/Image Processing/brain_tumor_dataset", recursive = T)
print(base_dir)
folder1<-"D:/vit/Image Processing/brain_tumor_dataset/yes"
folder2<-"D:/vit/Image Processing/brain_tumor_dataset/no"
file1<-list.files(folder1, pattern=NULL, all.files=FALSE,full.names=FALSE)
length(file1)
file2<-list.files(folder2, pattern=NULL, all.files=FALSE,full.names=FALSE)
length(file2)

library(imager)
#plot sample yes image
im <- load.image("D:/vit/Image Processing/brain_tumor_dataset/yes/Y1.jpg")
plot(im)
img <- load.image("D:/vit/Image Processing/brain_tumor_dataset/no/30 no.jpg")
plot(img)


# Load necessary libraries
library(keras)
library(tensorflow)
library(reticulate)
os<-import("os")
# Load the Brain tumor detection dataset
data_dir <- "D:/vit/Image Processing/brain_tumor_dataset"
setwd("D:/vit/Image Processing/brain_tumor_dataset")
files <- os$listdir(getwd())
dir_path<-getwd()


# Load necessary libraries
library(jpeg)
library(imager)


# Define the training, validation, and testing split ratios
train_split <- 0.7
val_split <- 0.15
test_split <- 0.15

# Use the image_dataset_from_directory function to split the data
image_dataset <- image_dataset_from_directory(
  directory = dir_path,
  validation_split = val_split + test_split,
  subset = "training",
  seed = 123,
  image_size = c(256, 256),
  batch_size = 32
)

validation_dataset <- image_dataset_from_directory(
  directory = dir_path,
  validation_split = val_split,
  subset = "validation",
  seed = 123,
  image_size = c(256, 256),
  batch_size = 32
)

test_dataset <- image_dataset_from_directory(
  directory = dir_path,
  validation_split = test_split,
  subset = "validation",
  seed = 123,
  image_size = c(256, 256),
  batch_size = 32
)







#VGG19 MODEL
library(keras)
# create the base pre-trained model
base_model1 <- application_vgg19( include_top = FALSE, weights = 'imagenet', input_shape=c(256,256,3))

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional  layers
for (layer in base_model1$layers)
  layer$trainable <- FALSE

# add our custom layers
predictions <- base_model1$output %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(0.25) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# this is the model we will train
model1 <- keras_model(inputs = base_model1$input, outputs = predictions)
model1
# Compile the model
model1 %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
# Train the model
acc_values_vgg19 <- c()
history <- model1 %>% fit(
  image_dataset,
  epochs = 10,
  validation_data = validation_dataset,
  callbacks = list(
    callback_lambda(
      on_epoch_end = function(epoch, logs) {
        acc_values_vgg19 <<- c(acc_values_vgg19, logs$acc)
      }
    )
  )
)
print(acc_values_vgg19)

#XCEPTION MODEL
library(keras)
base_model2 <- application_xception(
  weights = 'imagenet', # Load weights pre-trained on ImageNet.
  input_shape = c(256, 256, 3),
  include_top = FALSE # Do not include the ImageNet classifier at the top.
)
for (layer in base_model2$layers)
  layer$trainable <- FALSE

predictions <- base_model2$output %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(0.25) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model2 <- keras_model(inputs = base_model2$input, outputs = predictions)
model2
model2 %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
acc_values_xception=c()
history <- model2 %>% fit(
  image_dataset,
  epochs = 10,
  validation_data = validation_dataset,
  callbacks = list(
    callback_lambda(
      on_epoch_end = function(epoch, logs) {
        acc_values_xception <<- c(acc_values_xception, logs$acc)
      }
    )
  )
)
print(acc_values_xception)

##  RESNET50 MODEL
library(keras)
base_model3 <- application_resnet50(
  weights = 'imagenet', # Load weights pre-trained on ImageNet.
  input_shape = c(256, 256, 3),
  include_top = FALSE # Do not include the ImageNet classifier at the top.
)
for (layer in base_model3$layers)
  layer$trainable <- FALSE

predictions <- base_model3$output %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(0.25) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model3 <- keras_model(inputs = base_model3$input, outputs = predictions)
model3
model3 %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
acc_values_resnet=c()
history <- model3 %>% fit(
  image_dataset,
  epochs = 10,
  validation_data = validation_dataset,
  callbacks = list(
    callback_lambda(
      on_epoch_end = function(epoch, logs) {
        acc_values_resnet <<- c(acc_values_resnet, logs$acc)
      }
    )
  )
  )
print(acc_values_resnet)
### DenseNet

library(keras)
base_model4 <- application_densenet121(
  weights = 'imagenet', # Load weights pre-trained on ImageNet.
  input_shape = c(256, 256, 3),
  include_top = FALSE # Do not include the ImageNet classifier at the top.
)
for (layer in base_model4$layers)
  layer$trainable <- FALSE

predictions <- base_model4$output %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(0.25) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model4 <- keras_model(inputs = base_model4$input, outputs = predictions)
model4
model4 %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
acc_values_densenet=c()
history <- model4 %>% fit(
  image_dataset,
  epochs = 10,
  validation_data = validation_dataset,
  callbacks = list(
    callback_lambda(
      on_epoch_end = function(epoch, logs) {
        acc_values_densenet<<- c(acc_values_densenet, logs$acc)
      }
    )
  )
)
print(acc_values_densenet)
# Evaluate the model on the testing dataset
model4 %>% evaluate(test_dataset)

#mobilenet

library(keras)
base_model5 <- application_mobilenet(
  weights = 'imagenet', # Load weights pre-trained on ImageNet.
  input_shape = c(256, 256, 3),
  include_top = FALSE # Do not include the ImageNet classifier at the top.
)

for (layer in base_model5$layers)
  layer$trainable <- FALSE

predictions <- base_model5$output %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(0.25) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model5 <- keras_model(inputs = base_model5$input, outputs = predictions)
model5
model5 %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
acc_values_mobilenet <- c()
history <- model5 %>% fit(
  image_dataset,
  epochs = 10,
  validation_data = validation_dataset,
  callbacks = list(
    callback_lambda(
      on_epoch_end = function(epoch, logs) {
        acc_values_mobilenet <<- c(acc_values_mobilenet, logs$acc)
      }
    )
  )
  )
print(acc_values_mobilenet)
# Evaluate the model on the testing dataset
model5 %>% evaluate(test_dataset)


df<-data.frame(acc_values_vgg19,acc_values_xception,acc_values_resnet,acc_values_densenet,acc_values_mobilenet)
