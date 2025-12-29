"""
FIXED CELLS FOR jamal-ai-metric-learning.ipynb
==============================================

Copy these cells to your Kaggle notebook to replace the Lambda layers
with proper custom Layer classes that serialize correctly.

Instructions:
1. Replace CELL 7 (Build Siamese Model) with the code below
2. Train the model as usual
3. Run the export_for_deployment() function at the end
"""

# =============================================================================
# REPLACEMENT FOR CELL 7: BUILD SIAMESE MODEL (WITH PROPER CUSTOM LAYERS)
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow.keras.backend as K


# --- CUSTOM LAYERS (Serializable - replaces Lambda) ---

@tf.keras.utils.register_keras_serializable(package='JamalAI')
class L2NormalizeLayer(layers.Layer):
    """L2 Normalization layer that serializes properly."""

    def __init__(self, axis=1, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.keras.backend.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='JamalAI')
class EuclideanDistanceLayer(layers.Layer):
    """Euclidean distance layer for Siamese networks."""

    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        featsA, featsB = inputs
        sum_squared = tf.reduce_sum(tf.square(featsA - featsB), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

    def get_config(self):
        return super().get_config()


# --- CONTRASTIVE LOSS (unchanged) ---
MARGIN = 1.0

def contrastive_loss(y_true, y_pred):
    """Contrastive Loss Function for Metric Learning."""
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(MARGIN - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# --- BASE NETWORK (Using custom layers instead of Lambda) ---
def create_base_network():
    """
    Base network with proper custom layers (NOT Lambda).
    """
    input_seq = Input(shape=(MAX_LEN,), name='input_sequence')

    # Embedding layer
    x = layers.Embedding(MAX_VOCAB, EMBEDDING_DIM, name='embedding')(input_seq)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=False), name='bilstm')(x)

    # Dropout
    x = layers.Dropout(0.3)(x)

    # Dense layer
    x = layers.Dense(DENSE_UNITS, activation='relu', name='dense')(x)

    # L2 Normalize using CUSTOM LAYER (not Lambda!)
    x = L2NormalizeLayer(axis=1, name='l2_norm')(x)

    return Model(input_seq, x, name='base_network')


# Create base network
base_network = create_base_network()
print("🔧 BASE NETWORK ARCHITECTURE:")
base_network.summary()

# --- SIAMESE MODEL ---
input_a = Input(shape=(MAX_LEN,), name='input_sentence_1')
input_b = Input(shape=(MAX_LEN,), name='input_sentence_2')

# Share weights
embedding_a = base_network(input_a)
embedding_b = base_network(input_b)

# Compute distance using CUSTOM LAYER (not Lambda!)
distance = EuclideanDistanceLayer(name='euclidean_distance')([embedding_a, embedding_b])

# Final model
siamese_model = Model(inputs=[input_a, input_b], outputs=distance, name='siamese_network')
siamese_model.compile(loss=contrastive_loss, optimizer='adam')

print("\n🔧 SIAMESE NETWORK ARCHITECTURE:")
siamese_model.summary()


# =============================================================================
# EXPORT FUNCTION (Add at end of notebook)
# =============================================================================

def export_for_deployment():
    """Export model in SavedModel format for deployment."""
    import os
    import pickle

    os.makedirs('export', exist_ok=True)

    # Save as SavedModel format
    model_path = 'export/jamal_model'
    siamese_model.save(model_path, save_format='tf')
    print(f"✅ Model saved to {model_path}/")

    # Save tokenizer
    with open('export/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("✅ Tokenizer saved to export/tokenizer.pkl")

    # Save config
    config = {
        'max_len': MAX_LEN,
        'max_vocab': MAX_VOCAB,
        'margin': MARGIN,
        'best_threshold': best_threshold
    }
    with open('export/config.pkl', 'wb') as f:
        pickle.dump(config, f)
    print("✅ Config saved to export/config.pkl")

    print("\n📦 Download the 'export/' folder")
    print("   Use: !zip -r export.zip export/")


# Run after training:
# export_for_deployment()
