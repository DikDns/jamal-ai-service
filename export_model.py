"""
Script to export trained model and tokenizer for deployment.
Run this after training the model in Kaggle notebook.

IMPORTANT: Copy this entire code to the END of your jamal-ai-metric-learning.ipynb notebook
and run it after training completes.
"""

import pickle
import os


def export_for_deployment():
    """
    Export model and tokenizer for FastAPI deployment.
    This function assumes these variables exist from training:
    - siamese_model: The trained Siamese network
    - tokenizer: The Keras tokenizer
    - MAX_LEN, MAX_VOCAB, MARGIN, best_threshold: Training config
    """

    # Create output directory
    os.makedirs('export', exist_ok=True)

    # ========================================
    # SAVE MODEL IN SAVEDMODEL FORMAT
    # ========================================
    # SavedModel format is more portable than H5 for Lambda layers
    model_path = 'export/jamal_model'
    siamese_model.save(model_path, save_format='tf')
    print(f"✅ Model saved to {model_path}/ (SavedModel format)")

    # ========================================
    # SAVE TOKENIZER
    # ========================================
    tokenizer_path = 'export/tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved to {tokenizer_path}")

    # ========================================
    # SAVE CONFIG
    # ========================================
    config = {
        'max_len': MAX_LEN,
        'max_vocab': MAX_VOCAB,
        'margin': MARGIN,
        'best_threshold': best_threshold
    }
    config_path = 'export/config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✅ Config saved to {config_path}")

    # ========================================
    # DOWNLOAD INSTRUCTIONS
    # ========================================
    print("\n" + "="*50)
    print("📦 DOWNLOAD THESE FILES FROM 'export/' FOLDER:")
    print("="*50)
    print("   1. jamal_model/  (folder - download entire folder)")
    print("      - saved_model.pb")
    print("      - variables/")
    print("      - assets/")
    print("   2. tokenizer.pkl")
    print("   3. config.pkl")
    print("\n💡 TIP: In Kaggle, right-click the 'export' folder and 'Download'")
    print("   Or use: !zip -r export.zip export/")


# ========================================
# RUN THIS IN YOUR NOTEBOOK:
# ========================================
# Uncomment and run after training:
# export_for_deployment()
