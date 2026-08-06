from src.models.qwen import Qwen25VLWrapper, QwenConfig


def test_qwen_wrapper_imports_and_initializes_without_loading_checkpoint():
    wrapper = Qwen25VLWrapper(QwenConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct"))
    assert wrapper.model_name == "qwen2.5-vl-7b-instruct"
    assert wrapper.config.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert wrapper._model is None
    assert wrapper._processor is None
