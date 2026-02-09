import logging
from model.predrnn_inference import PredRNNInference

logger = logging.getLogger("uvicorn.error")

predrnn_model = None  # shared variable


def load_predrnn_model():
    global predrnn_model

    if predrnn_model is None:
        logger.info("Loading PredRNN model...")
        predrnn_model = PredRNNInference(
            checkpoint_path="checkpoints/predrnn_best.pt",
            out_steps=6
        )
        logger.info("PredRNN model loaded successfully")

    return predrnn_model
