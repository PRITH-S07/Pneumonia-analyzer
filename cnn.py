from keras import regularizers
inputs = tf.keras.layers.Input((200,200,3))
s = tf.keras.layers.Lambda(lambda x: x/ 255)(inputs)

c1=tf.keras.layers.Conv2D(16, (3, 3),activation='relu')(s)
p1 = tf.keras.layers.MaxPooling2D(2, 2)(c1)

c2=tf.keras.layers.Conv2D(32, (3, 3),activation='relu')(p1)
p2 = tf.keras.layers.MaxPooling2D(2,2)(c2)

c3=tf.keras.layers.Conv2D(64, (3, 3),activation='relu')(p2)
p3 = tf.keras.layers.MaxPooling2D(2,2)(c3)

c4=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(p3)
p4 = tf.keras.layers.MaxPooling2D(2,2)(c4)

c5=tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(p4)
p5 = tf.keras.layers.MaxPooling2D(2,2)(c5)

c5=tf.keras.layers.Flatten()(p4)
c5=tf.keras.layers.Dense(512,activation='relu')(c5)
outputs = tf.keras.layers.Dense(1,activation='sigmoid')(c5)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

from tensorflow.keras.optimizers import RMSprop,Adam
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

