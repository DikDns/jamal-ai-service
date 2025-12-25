"""
Script to export trained model and tokenizer for deployment.
Run this after training the model in Kaggle notebook.
"""

import pickle
import os

# This script should be added at the END of your Kaggle notebook
# Run it after training to save the model and tokenizer


def export_for_deployment():
    """
    Export model and tokenizer for FastAPI deployment.
    Add this cell at the end of jamal-ai-metric-learning.ipynb
    """

    # Create output directory
    os.makedirs('export', exist_ok=True)

    # Save model (H5 format for compatibility)
    siamese_model.save('export/jamal_metric_learning.h5')
    print("✅ Model saved to export/jamal_metric_learning.h5")

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

    print("\n📦 Download files from 'export/' folder:")
    print("   - jamal_metric_learning.h5")
    print("   - tokenizer.pkl")
    print("   - config.pkl")

# Uncomment and run:
# export_for_deployment()
