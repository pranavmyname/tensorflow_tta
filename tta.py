import tensorflow as tf

def get_aug_img_v1(image):
    """ Get augmented image

    Args:
        image (tf.tensor): Tensor of type tf.float32 of shape BxHxWxC

    Returns:
        _type_: Tensor of type tf.float32 of shape B*5xHxWxC
    """
    aug_images = tf.concat([image,
                            tf.image.resize(tf.image.resize(image, [500,500]), [512, 512]),
                            tf.image.rot90(image, 1),
                            tf.image.flip_up_down(image),
                            tf.image.flip_left_right(image)
                            ],
                            axis = 0
                            )
    return aug_images

def get_deaug_images_v1(image):
    """Get deaugmented mask or image

    Args:
        image (tf.tensor): Model output goes here. It is of the shape B*5xHxWxC' where C' is the number
        of channels in the model output

    Returns:
        tf.tensor: Shape BxHxWxC'
    """
    i1, i2, i3, i4, i5 = tf.split(image, num_or_size_splits=5, axis=0)
    deaug_img = tf.stack([i1,
                           i2,
                           tf.image.rot90(i3, -1),
                           tf.image.flip_up_down(i4),
                           tf.image.flip_left_right(i5)
                           ],
                           axis = 0)
    out = tf.math.reduce_mean(deaug_img, axis = 0)
    return out

def infer_with_tta(model, input):
    """ Get output after performing test time augmentation

    Args:
        model (keras model): _description_
        input (tf.tensor): Tensor of shape BxHxWxC

    Returns:
        tf.tensor: Model output 
    """
    return get_deaug_images_v1(model(get_aug_img_v1(input)))