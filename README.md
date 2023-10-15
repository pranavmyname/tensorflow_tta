# tensorflow-tta
Test time augmentation on Tensorflow models. Inspired from https://github.com/BloodAxe/pytorch-toolbelt for pytorch.
To run the test time augmentation, simply run the following code:

```python
import tta
out = tta.infer_with_tta(model, input)
```
