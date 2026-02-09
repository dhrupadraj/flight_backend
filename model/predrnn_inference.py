import torch
import numpy as np
from model.predrnn import PredRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredRNNInference:
    def __init__(self, checkpoint_path: str, out_steps: int):
        self.model = PredRNN(
            input_dim=2,
            hidden_dims=[64, 64, 64],
            kernel_size=3,
            num_layers=3,
            out_steps=out_steps
        ).to(DEVICE)

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        input_sequence: (Tin, 2, H, W)
        returns: (Tout, 2, H, W)
        """
        x = torch.from_numpy(input_sequence).float().unsqueeze(0).to(DEVICE)
        preds = self.model(x)
        return preds.squeeze(0).cpu().numpy()
    def wind_to_cost(self,u: np.ndarray, v: np.ndarray) -> np.ndarray:
    
        wind_speed = np.sqrt(u**2 + v**2)
        return wind_speed  # higher wind = higher cost
