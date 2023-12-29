import onnxruntime
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class OnnxWrapper(nn.Module):
    def __init__(
        self,
        optimization_level,
        onnx_model_path: str,
        disabeled_optimizers: list = [],
        providers: str = ["CUDAExecutionProvider", "CPUExecutionProvider"],
        severity=None,
    ) -> None:
        super().__init__()
        
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = optimization_level
        self.onnxruntime_session = onnxruntime.InferenceSession(onnx_model_path,\
                                                                providers=providers,\
                                                                sess_options=sess_options,\
                                                                disabled_optimizers=disabeled_optimizers
                                                                )
        
        self.dict_adapter = dict

        if severity is not None:
            onnxruntime.set_default_logger_severity(severity)

    def forward(self, x: Tensor, xlen: Tensor) -> dict[str, Tensor]:
        """Forward method for encoder.
        @param x: (B, C, T)
        @type x: FloatTensor
        @param xlen: (B)
        @type xlen: Tensor, range - [0, 1]
        @return: logits - (B, C, T), log_probs - (B, C, T), olen - (B), uncertainty - (B)
        @rtype: Dict[str, Tensor].
        """
        logits = self.onnxruntime_session.run(None, {"x": x.squeeze(1).cpu().numpy(), "xlen": xlen.cpu().numpy()})[0]
        logits = torch.as_tensor(logits, device=x.device)
        log_probs = F.log_softmax(logits, dim=1)

        return self.dict_adapter(logits=logits, log_probs=log_probs)