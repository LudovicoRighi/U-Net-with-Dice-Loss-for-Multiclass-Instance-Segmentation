# U-Net with Dice Loss for Multiclass Instance Segmentation

In this repository you can find all the material used to take part at the competitions created for the Artifical Neural Networks and Deep Learning exam at Politecnico di Milano.
The goal of this challenge is to build the best model to solve a segmentation problem. The model should learn how to segment Crop, Weed and Background at pixel level in order to optimize the agricultural processes and to reduce the use of resources (water, fertilizers, and pesticides).
 
<p align="left">
  <img width="46%"  src="https://user-images.githubusercontent.com/52406034/135988327-850399b7-d176-45c7-927a-0f8ef2de0f27.png">
  <img width="46%" hspace="3%" src="https://user-images.githubusercontent.com/52406034/135988570-98bb2e95-c387-4c32-822e-1485a4270413.png">
</p> 


## Our Best Model

The model that better performed in our competition was a custom implementation of a U-Net. An interesting problem to solve was the unbalanced frequency and size of background, crop and weed in our dataset images. In order to overcome this situation we tried to exploit different loss functions: Cathegorical Focal Loss Function and Multiclass Dice Loss.


### Categorical Focal Loss

The Focal loss adds a factor (1pt)^γ to the standard cross entropy criterion. Setting γ > 0 reduces the relative loss for well-classified examples (pt > .5), putting more focus on hard, misclassified examples.
We try to add this loss to our models, but we did not get significant improvement with respect to the classical categorical cross entropy.

### Multiclass Dice Loss

In cross entropy loss, the loss is calculated as the average of per-pixel loss, and the per-pixel loss is calculated discretely, without knowing whether its adjacent pixels are boundaries or not. As a result, cross entropy loss only considers loss in a micro sense rather than considering it globally, which is not enough for image level prediction.
So we decided to implement our multiclass version of the Dice-Loss: a loss that considers the loss information both locally and globally.
With the dice-loss we get an average boost in the meanIoU from 1 to 5 per cent in our unet models.

## Conclusion

For a more in-depth analysis I recommend you to read the report of the challenge. The final evaluation of this project by our professor was 2.0/2.0.
