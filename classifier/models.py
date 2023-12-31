from django.db import models
import tensorflow as tf
import datetime

class MnistModel(models.Model):
    
    def mnist_model(epochs, batch_size):
        """
        load mnist dataset from tensorflow
        """
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # divide with 255 to make pixels between 0 and 1
        x_train, x_test = x_train / 255.0, x_test / 255.0

        """
        build model and set optimizer
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        """
        train
        """
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

        """
        evaluate
        """
        model.evaluate(x_test, y_test, verbose=2)

        """
        export saved model
        """
        current_date = datetime.datetime.now().date().strftime('%Y%m%d')
        model_date = f'mnist/{current_date}'
        tf.keras.models.save_model(model, model_date)

    class Meta:
        app_label = 'classifier'  # 해당 모델이 속한 앱의 이름을 지정