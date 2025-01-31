import tensorflow as tf


def normalize_img(img):
  return (img / 127.5) - 1

def create_message_tensor_for_training(width, height, data_depth):
  message = tf.random.uniform([width, height, data_depth], 0, 2, dtype=tf.int32)
  message = tf.cast(message, tf.float32)
  return message

def create_message_dataset(dataset_length, width, height, data_depth):
  message_tensors = [create_message_tensor_for_training(width, height, data_depth) for _ in range(dataset_length)]
  return tf.data.Dataset.from_tensor_slices(message_tensors)


# @tf.function
# def create_binary_tensor_batch(self, cover_image, data_depth):
#   batch_size = tf.shape(cover_image)[0]
#   _, height, width, _ = cover_image.shape

#   random_tensor = tf.random.uniform(shape=(batch_size, height, width, data_depth), minval=0, maxval=2, dtype=tf.int32)
#   message = tf.cast(random_tensor, tf.float32)

#   return message