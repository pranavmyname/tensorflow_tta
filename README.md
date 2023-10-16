# tensorflow-tta
**Tensorflow Test Time Augmentation**: Test time augmentation on Tensorflow keras models which utilizes GPU acceleration for segmentation models. Inspired from https://github.com/BloodAxe/pytorch-toolbelt for pytorch.
To run the test time augmentation, simply run the following code:

```python
import tta
out = tta.infer_with_tta(model, input)
```

To create keras augmentation layers and add it in the model itself for GPU acceleration, an example code has been pasted below:

```python
import tta
import tensorflow as tf
aug_layer = tta.TTAug()
deaug_layer = tta.TTDeAug()
x = tf.keras.layers.Input(shape = (None, None, model.input.shape[-1]))
x_aug = aug_layer(x)
out = model(aug)
out_deaug = deaug_layer(out)
tta_model = tf.keras.Model(inputs = x, outputs = output_deaug)
```
