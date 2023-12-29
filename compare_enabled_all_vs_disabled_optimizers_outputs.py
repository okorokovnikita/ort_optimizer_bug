import onnx
import onnxruntime
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import matplotlib.pyplot as plt

torch.manual_seed(1337)

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
        self.onnxruntime_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=providers,
            sess_options=sess_options,
            disabled_optimizers=disabeled_optimizers,
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
        logits = self.onnxruntime_session.run(
            None, {"x": x.squeeze(1).cpu().numpy(), "xlen": xlen.cpu().numpy()}
        )[0]
        logits = torch.as_tensor(logits, device=x.device)
        log_probs = F.log_softmax(logits, dim=1)

        return self.dict_adapter(logits=logits, log_probs=log_probs)


def compute_onnx_logits(wrapper, xlen, input):
    wrapper_out = wrapper.forward(input, xlen)
    wrapper_logits = wrapper_out["logits"]

    return wrapper_logits


def main():
    INPUT_TENSOR_PATH = "/mnt/srs-speechcore-data/tmp/ort_issue/input_tensor.pt"
    INPUT_LENGTH_PATH = "/mnt/srs-speechcore-data/tmp/ort_issue/length_tensor.pt"
    ONNX_MODEL_PATH = "/mnt/srs-speechcore-data/tmp/ort_issue/model_onnx.onnx"

    input_tensor = torch.load(INPUT_TENSOR_PATH)
    input_length = torch.load(INPUT_LENGTH_PATH)

    wrapper_enable_all = OnnxWrapper(
        onnx_model_path=ONNX_MODEL_PATH,
        optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
    )

    onnx_logits_enable_all = compute_onnx_logits(
        wrapper_enable_all, input_length, input_tensor
    )

    wrapper_enable_extended = OnnxWrapper(
        onnx_model_path=ONNX_MODEL_PATH,
        optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    )

    onnx_logits_enable_extended = compute_onnx_logits(
        wrapper_enable_extended, input_length, input_tensor
    )

    wrapper_disable_all = OnnxWrapper(
        onnx_model_path=ONNX_MODEL_PATH,
        optimization_level=onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL,
    )

    onnx_logits_disable_all = compute_onnx_logits(
        wrapper_disable_all, input_length, input_tensor
    )

    wrapper_disable_layernormfusion_simplifiedlayernormfusion = OnnxWrapper(
        onnx_model_path=ONNX_MODEL_PATH,
        optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
        disabeled_optimizers=["LayerNormFusion", "SimplifiedLayerNormFusion"],
    )

    onnx_logits_disable_layernormfusion_simplifiedlayernormfusion = compute_onnx_logits(
        wrapper_disable_layernormfusion_simplifiedlayernormfusion,
        input_length,
        input_tensor,
    )

    wrapper_disable_layernormfusion = OnnxWrapper(
        onnx_model_path=ONNX_MODEL_PATH,
        optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
        disabeled_optimizers=["LayerNormFusion"],
    )

    onnx_logits_disable_layernormfusion = compute_onnx_logits(
        wrapper_disable_layernormfusion, input_length, input_tensor
    )

    wrapper_disable_simplifiedlayernormfusion = OnnxWrapper(
        onnx_model_path=ONNX_MODEL_PATH,
        optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
        disabeled_optimizers=["SimplifiedLayerNormFusion"],
    )

    onnx_logits_disable_simplifiedlayernormfusion = compute_onnx_logits(
        wrapper_disable_simplifiedlayernormfusion, input_length, input_tensor
    )

    print(
        f"Outputs of model with ort.GraphOptimizationLevel.ORT_ENABLE_ALL are the same as with ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED? {torch.equal(onnx_logits_enable_all, onnx_logits_enable_extended)}"
    )
    print("\n")

    print(
        f"Outputs of model with ort.GraphOptimizationLevel.ORT_ENABLE_ALL are the same as with ort.GraphOptimizationLevel.DISABLE_ALL? {torch.equal(onnx_logits_enable_all, onnx_logits_disable_all)}"
    )
    print(
        f"Max absolute diff: {torch.max(torch.abs(onnx_logits_enable_all - onnx_logits_disable_all)).item()}"
    )
    print(
        f"Mean absolute diff: {torch.mean(torch.abs(onnx_logits_enable_all - onnx_logits_disable_all)).item()}"
    )
    print("\n")

    print(
        f"Outputs of model with ort.GraphOptimizationLevel.ORT_ENABLE_ALL are the same as with disabled LayerNormFusion and SimplifiedLayerNormFusion? {torch.equal(onnx_logits_enable_all, onnx_logits_disable_layernormfusion_simplifiedlayernormfusion)}"
    )
    print(
        f"Max absolute diff: {torch.max(torch.abs(onnx_logits_enable_all - onnx_logits_disable_layernormfusion_simplifiedlayernormfusion)).item()}"
    )
    print(
        f"Mean absolute diff: {torch.mean(torch.abs(onnx_logits_enable_all - onnx_logits_disable_layernormfusion_simplifiedlayernormfusion)).item()}"
    )
    print("\n")

    print(
        f"Outputs of model with ort.GraphOptimizationLevel.ORT_ENABLE_ALL are the same as with disabled LayerNormFusion? {torch.equal(onnx_logits_enable_all, onnx_logits_disable_layernormfusion)}"
    )
    print(
        f"Max absolute diff: {torch.max(torch.abs(onnx_logits_enable_all - onnx_logits_disable_layernormfusion)).item()}"
    )
    print(
        f"Mean absolute diff: {torch.mean(torch.abs(onnx_logits_enable_all - onnx_logits_disable_layernormfusion)).item()}"
    )
    print("\n")

    print(
        f"Outputs of model with ort.GraphOptimizationLevel.ORT_ENABLE_ALL are the same as with disabled SimplifiedLayerNormFusion? {torch.equal(onnx_logits_enable_all, onnx_logits_disable_simplifiedlayernormfusion)}"
    )
    print(
        f"Max absolute diff: {torch.max(torch.abs(onnx_logits_enable_all - onnx_logits_disable_simplifiedlayernormfusion)).item()}"
    )
    print(
        f"Mean absolute diff: {torch.mean(torch.abs(onnx_logits_enable_all - onnx_logits_disable_simplifiedlayernormfusion)).item()}"
    )
    print("\n")

if __name__ == "__main__":
    main()
