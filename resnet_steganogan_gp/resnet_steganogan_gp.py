import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from imageio.v2 import imread, imwrite
from utils import text_to_bits, bits_to_text


class ResnetSteganoGAN(tf.keras.Model):
  def __init__(self, encoder=None, decoder=None, critic=None, data_depth=6, critic_extra_steps=5, gp_weight=10.0):
    super(ResnetSteganoGAN, self).__init__()

    self.data_depth = data_depth

    self.encoder = encoder # ResidualEncoder(data_depth)
    self.decoder = decoder # BasicDecoder(data_depth)
    self.critic  = critic  # BasicCritic()

    self.critic_extra_steps = critic_extra_steps
    self.gp_weight = gp_weight

    self.encoder_decoder_total_loss_tracker = Mean(name="encoder_decoder_total_loss")
    self.critic_loss_tracker                = Mean(name="critic_loss")
    self.similarity_loss_tracker            = Mean(name="similarity_loss")
    self.decoder_loss_tracker               = Mean(name="decoder_loss")
    self.decoder_accuracy_tracker           = Mean(name="decoder_accuracy")
    self.realism_loss_tracker               = Mean(name="realism_loss")
    self.psnr_tracker                       = Mean(name="psnr")
    self.ssim_tracker                       = Mean(name="ssim")
    self.gradient_penalty_tracker           = Mean(name="gradient_penalty")

  def compile(self, encoder_optimizer, decoder_optimizer, critic_optimizer, similarity_loss_fn, decoder_loss_fn):
    super(ResnetSteganoGAN, self).compile()
    self.encoder_optimizer   = encoder_optimizer   # Adam(learning_rate=1e-4)
    self.decoder_optimizer   = decoder_optimizer   # Adam(learning_rate=1e-4)
    self.critic_optimizer    = critic_optimizer    # Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    self.similarity_loss_fn  = similarity_loss_fn  # MeanSquaredError()
    self.decoder_loss_fn     = decoder_loss_fn     # BinaryCrossentropy(from_logits=False)

  @property
  def metrics(self):
    return [
      self.encoder_decoder_total_loss_tracker,
      self.critic_loss_tracker,
      self.similarity_loss_tracker,
      self.decoder_loss_tracker,
      self.decoder_accuracy_tracker,
      self.realism_loss_tracker,
      self.psnr_tracker,
      self.ssim_tracker,
      self.gradient_penalty_tracker
    ]

  def decoder_accuracy(self, messages, recovered_messages):
    binary_payloads = tf.greater_equal(messages, 0.5)
    binary_decodeds = tf.greater_equal(recovered_messages, 0.5)

    equal_elements = tf.equal(binary_payloads, binary_decodeds)
    casted_elements = tf.cast(equal_elements, tf.float32)
    decoder_accuracy = tf.reduce_mean(casted_elements)

    return decoder_accuracy

  def critic_loss_fn(self, cover_critic_score, stego_critic_score):
    cover_critic_score = tf.reduce_mean(cover_critic_score)
    stego_critic_score = tf.reduce_mean(stego_critic_score)

    return stego_critic_score - cover_critic_score # fake - real

  def realism_loss_fn(self, stego_critic_score):
    return -tf.reduce_mean(stego_critic_score)

  def gradient_penalty(self, batch_size, real_images, fake_images, training=True):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
      gp_tape.watch(interpolated)
      pred = self.critic(interpolated, training=training)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

  @tf.function
  def train_step(self, data):
    cover_images, messages = data
    batch_size = tf.shape(cover_images)[0]

    for _ in range(self.critic_extra_steps):
      with tf.GradientTape() as critic_tape:
        stego_images = self.encoder([cover_images, messages], training=True)

        cover_critic_scores = self.critic(cover_images, training=True)
        stego_critic_scores = self.critic(stego_images, training=True)

        gp = self.gradient_penalty(batch_size, cover_images, stego_images)
        critic_cost = self.critic_loss_fn(cover_critic_scores, stego_critic_scores)
        critic_loss = critic_cost + gp * self.gp_weight

      critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
      self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))


    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
      stego_images = self.encoder([cover_images, messages], training=True)
      stego_critic_scores = self.critic(stego_images, training=True)
      recovered_messages = self.decoder(stego_images, training=True)

      decoder_accuracy = self.decoder_accuracy(messages, recovered_messages)

      similarity_loss = self.similarity_loss_fn(cover_images, stego_images)
      decoder_loss = self.decoder_loss_fn(messages, recovered_messages)
      realism_loss = self.realism_loss_fn(stego_critic_scores)
      encoder_decoder_total_loss = similarity_loss + decoder_loss + realism_loss

    encoder_grads = encoder_tape.gradient(encoder_decoder_total_loss, self.encoder.trainable_variables)
    decoder_grads = decoder_tape.gradient(encoder_decoder_total_loss, self.decoder.trainable_variables)

    self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.encoder.trainable_variables))
    self.decoder_optimizer.apply_gradients(zip(decoder_grads, self.decoder.trainable_variables))

    self.encoder_decoder_total_loss_tracker.update_state(encoder_decoder_total_loss)
    self.critic_loss_tracker.update_state(critic_loss)
    self.similarity_loss_tracker.update_state(similarity_loss)
    self.decoder_loss_tracker.update_state(decoder_loss)
    self.decoder_accuracy_tracker.update_state(decoder_accuracy)
    self.realism_loss_tracker.update_state(realism_loss)
    self.psnr_tracker.update_state(tf.image.psnr(cover_images, stego_images, max_val=1.0))
    self.ssim_tracker.update_state(tf.image.ssim(cover_images, stego_images, max_val=1.0))
    self.gradient_penalty_tracker.update_state(gp)

    return {
      'encoder_decoder_total_loss': self.encoder_decoder_total_loss_tracker.result(),
      'critic_loss': self.critic_loss_tracker.result(),
      'similarity_loss': self.similarity_loss_tracker.result(),
      'decoder_loss': self.decoder_loss_tracker.result(),
      'decoder_accuracy': self.decoder_accuracy_tracker.result(),
      'realism_loss': self.realism_loss_tracker.result(),
      'psnr': self.psnr_tracker.result(),
      'ssim': self.ssim_tracker.result(),
      'gradient_penalty': self.gradient_penalty_tracker.result()
    }

  @tf.function 
  def test_step(self, data):
    cover_images, messages = data
    batch_size = tf.shape(cover_images)[0]

    stego_images = self.encoder([cover_images, messages], training=False)
    stego_critic_scores = self.critic(stego_images, training=False)
    recovered_messages = self.decoder(stego_images, training=False)

    decoder_accuracy = self.decoder_accuracy(messages, recovered_messages)

    similarity_loss = self.similarity_loss_fn(cover_images, stego_images)
    decoder_loss = self.decoder_loss_fn(messages, recovered_messages)
    realism_loss = self.realism_loss_fn(stego_critic_scores)
    encoder_decoder_total_loss = similarity_loss + decoder_loss + realism_loss

    cover_critic_scores = self.critic(cover_images, training=False)
    stego_critic_scores = self.critic(stego_images, training=False)

    gp = self.gradient_penalty(batch_size, cover_images, stego_images, training=False)
    critic_cost = self.critic_loss_fn(cover_critic_scores, stego_critic_scores)
    critic_loss = critic_cost + gp * self.gp_weight

    self.encoder_decoder_total_loss_tracker.update_state(encoder_decoder_total_loss)
    self.critic_loss_tracker.update_state(critic_loss)
    self.similarity_loss_tracker.update_state(similarity_loss)
    self.decoder_loss_tracker.update_state(decoder_loss)
    self.decoder_accuracy_tracker.update_state(decoder_accuracy)
    self.realism_loss_tracker.update_state(realism_loss)
    self.psnr_tracker.update_state(tf.image.psnr(cover_images, stego_images, max_val=1.0))
    self.ssim_tracker.update_state(tf.image.ssim(cover_images, stego_images, max_val=1.0))
    self.gradient_penalty_tracker.update_state(gp)

    return {
      'encoder_decoder_total_loss': self.encoder_decoder_total_loss_tracker.result(),
      'critic_loss': self.critic_loss_tracker.result(),
      'similarity_loss': self.similarity_loss_tracker.result(),
      'decoder_loss': self.decoder_loss_tracker.result(),
      'decoder_accuracy': self.decoder_accuracy_tracker.result(),
      'realism_loss': self.realism_loss_tracker.result(),
      'psnr': self.psnr_tracker.result(),
      'ssim': self.ssim_tracker.result(),
      'gradient_penalty': self.gradient_penalty_tracker.result()
    }

  @tf.function
  def call(self, data):
    cover_images, messages = data
    
    stego_images = self.encoder([cover_images, messages], training=False)
    recovered_messages = self.decoder(stego_images, training=False)

    return stego_images, recovered_messages

  def _image_to_tensor(self, image):
    image = tf.cast(image, tf.float32)
    image = tf.convert_to_tensor(image)
    image = (image / 127.5) - 1.0 # Normalize to [-1, 1]
    image = tf.expand_dims(image, axis=0)
    return image

  def _stego_tensor_to_image(self, stego_tensor):
    stego_image = tf.squeeze(stego_tensor)
    stego_image = (stego_image + 1.0) * 127.5 # Denormalize to [0, 255]
    stego_image = tf.cast(stego_image, tf.uint32)
    return stego_image

  def encode(self, cover_path, stego_path, message):
    cover = imread(cover_path)
    cover_tensor = self._image_to_tensor(cover)
    width, height, _ = cover.shape
    message_shape = (height, width, self.data_depth)

    message = text_to_bits(message, message_shape)
    message = np.reshape(message, (1, width, height, self.data_depth)) 
    message = tf.convert_to_tensor(message, dtype=tf.float32)

    stego_tensor = self.encoder([cover_tensor, message])
    stego_image = self._stego_tensor_to_image(stego_tensor)
    imwrite(stego_path, stego_image.numpy().astype(np.uint8))

  def decode(self, stego):
    stego = imread(stego)
    stego_tensor = self._image_to_tensor(stego)
    message_shape = (stego.shape[1], stego.shape[0], self.data_depth)

    message_tensor = self.decoder(stego_tensor)
    message_tensor = tf.round(message_tensor)
    message_tensor = tf.cast(message_tensor, tf.int8)

    message = bits_to_text(message_tensor, message_shape)
    return message