#Quantitize the onnx model for better performance
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def main(input_onnx_file, output_onnx_file):
    # Load the ONNX model
    model = onnx.load(input_onnx_file)

    # Quantize the ONNX model
    quantized_model = quantize_dynamic(
        input_onnx_file,
        output_onnx_file,
        weight_type=QuantType.QUInt8,
        reduce_range=True,
    )

    print(f"Quantized ONNX model saved to {output_onnx_file}")

if __name__ == "__main__":
    input_onnx_file = "Model/yolov4-tiny_224_v3.onnx"
    output_onnx_file = "Model/yolov4-tiny_224_v3_quant.onnx"

    main(input_onnx_file, output_onnx_file)
