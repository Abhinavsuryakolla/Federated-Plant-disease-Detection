import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def create_global_model():
    # Define MobileNetV2 architecture with ImageNet weights
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base layers initially
    base_model.trainable = False
    
    # Add custom head for 38 classes (or whatever is the actual number, let's assume 38 here, we will generalize but MobileNetV2 will adapt)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Using 33 classes since that's what was actually found in the dataset
    num_classes = 33 
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    model = create_global_model()
    model.save('models/global_weights_round0.h5')
    print("Global model initialized and saved to models/global_weights_round0.h5")
