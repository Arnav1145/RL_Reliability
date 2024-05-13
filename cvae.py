import numpy as np
import tensorflow as tf

from data_prep import DataPrep, Vec2Img


class CVAE(tf.keras.Model):
    """
    Conditional Variational Autoencoder;
    It provides methods for developing encoder/decoder
    """

    def __init__(self, 
                 latent_dim=2,
                 image_size=21,
                 label_dim=20210) -> None: # ! Changes - 1
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.label_dim = label_dim # ! Changes - 2

        # Losses: Total loss = reconstruction_loss + KL_loss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="Total loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="Reconstruction loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="KL loss")
        
        # ? See these changes for dynamic behaviour
        # self.label = tf.keras.Input(shape=(num_classes,))
        # x = layers.concatenate([x, label])
    
    def Sampling(self, inputs) -> np.ndarray:
        """
        Returns latent x using reparameterization trick (x = mu + sigma*epsilon)
        """
        x_mean, x_logvar = inputs
        batch = tf.shape(x_mean)[0]
        dim = tf.shape(x_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return x_mean + tf.exp(0.5 * x_logvar) * epsilon

    def Encoder(self, filters, kernel, stride, neurons) -> tf.keras.Model:

        encoder_inputs_x = tf.keras.Input(shape=(self.image_size,self.image_size,1))

         # ! Changes - 3 Start
        encoder_inputs_y = tf.keras.Input(shape=(self.label_dim,))
        encoder_inputs_y_reshaped = tf.keras.layers.Dense(self.image_size * self.image_size * 1)(encoder_inputs_y)
        encoder_inputs_y_reshaped = tf.keras.layers.Reshape((self.image_size, self.image_size, 1))(encoder_inputs_y_reshaped)
        encoder_inputs = tf.keras.layers.Concatenate(axis = -1)([encoder_inputs_x, encoder_inputs_y_reshaped])
         # ! Changes - 3 End

        # time_inputs = tf.keras.Input(shape=(1,))
        x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2D_layer_1")(encoder_inputs)
        x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2D_layer_2")(x)
        x = tf.keras.layers.Flatten(name="flattened")(x)
        # x = tf.keras.layers.concatenate([x,time_inputs], name="concatenate1") # ! Changes
        z = tf.keras.layers.Dense(neurons, activation='relu', name="dense_layer")(x)

        x_mean = tf.keras.layers.Dense(self.latent_dim, name="x_mean")(z)
        x_logvar = tf.keras.layers.Dense(self.latent_dim, name="x_logvar")(z)
        x = self.Sampling([x_mean, x_logvar])

        self.encoder = tf.keras.Model([encoder_inputs_x, encoder_inputs_y], [x_mean, x_logvar, x], name="encoder")
        self.encoder.summary()

        return self.encoder

    def Decoder(self, filters, kernel, stride, neurons) -> tf.keras.Model:

        op_filters, sens_filters = filters

        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))

        # ! Changes - 5 Start
        decoder_input_y = tf.keras.Input(shape=(self.label_dim,))
        decoder_inputs = tf.keras.layers.Concatenate()([latent_inputs, decoder_input_y])
        # ! Changes - 5 Start

        # ! Changes - 6 
        x = tf.keras.layers.Dense(self.encoder.get_layer('flattened').output_shape[1], activation='relu', name="latent_layer_image")(decoder_inputs)
        
        x = tf.keras.layers.Reshape(self.encoder.get_layer('conv2D_layer_2').output_shape[1:])(x)
        x = tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2DTranspose_layer_1")(x)
        x = tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2DTranspose_layer_2")(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=kernel, strides=1, activation='sigmoid', padding='same', name="conv2DTranspose_layer_out")(x) # ! Changes
        self.decoder = tf.keras.Model([latent_inputs, decoder_input_y], decoder_outputs, name="decoder")
        self.decoder.summary()

        return self.decoder

    @property
    def metrics(self) -> list:
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data) -> dict:
        with tf.GradientTape() as tape:
            x_mean, x_logvar, x = self.encoder(data)
            print(x_mean)
            print(x_logvar)
            print(x)
            reconstruction = self.decoder(x)
            print("data------------->",tf.concat([data[0][0], data[0][1]], axis = 1))  # ! Changes
            print("output--------------->",reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data[0][0], reconstruction), axis=(1,2)))
            kl_loss = -.5 * (1 + x_logvar - tf.square(x_mean) - tf.exp(x_logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

def img_norm(data):
    images = np.expand_dims(data, -1).astype("float32")

    return images

if __name__ == "__main__":

    file_path = "CMAPSSData/train_FD002.txt"
    num_settings = 3
    num_sensors = 21
    num_units = 100
    step = "VAE"

    filters = [8,16]
    kernel = 2
    stride = 2
    neurons = 4

    # Data prep
    data = DataPrep(file=file_path,
                    num_settings=num_settings, 
                    num_sensors=num_sensors, 
                    num_units=num_units, 
                    step=step)
    
    df = data.ReadData()
    # print(df)
    
    image_data = Vec2Img(df=df,
                         data=data,
                         image_size=num_settings+num_sensors,
                         plot=True)
    
    images = image_data.Transform()
    images = img_norm(images)
    
    # engine_lives = df.groupby(df['Unit']).size().tolist()
    # num_engines = len(engine_lives)
    # print(engine_lives)
    # print(num_engines)
    # print(len(df['NormTime']))

    y_train = tf.keras.utils.to_categorical(df['NormTime'], num_classes=(np.array(df["NormTime"]).shape)[0])
    # y_train = tf.cast(df.NormTime.values, dtype = tf.float64)
    # # y_train_shape = y_train.shape
    # y_train = tf.keras.layers.Reshape((None, (y_train.shape)[0]))

    n = CVAE(latent_dim=1,image_size=24)
    encoder = n.Encoder(filters, kernel, stride, neurons)
    decoder = n.Decoder(filters, kernel, stride, neurons)
    n.compile(optimizer=tf.keras.optimizers.Adam())
    n.fit([images, y_train], epochs=30, batch_size=48)

    # Save decoder to use later as RL environment
    decoder.save('saved_models/environment-cvae.h5')


