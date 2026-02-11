import logging

logger = logging.getLogger("uvicorn.error")

predrnn_model = None  # shared variable


def load_predrnn_model():
    global predrnn_model

    if predrnn_model is None:
        # Lazy import keeps service/module import fast and defers torch cost.
        from model.predrnn_inference import PredRNNInference

        logger.info("Loading PredRNN model...")
        predrnn_model = PredRNNInference(
            checkpoint_path="checkpoints/predrnn_best.pt",
            out_steps=6
        )
        logger.info("PredRNN model loaded successfully")

    return predrnn_model
