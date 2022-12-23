import launch

if not launch.is_installed("onnx"):
    launch.run_pip("install onnx", "requirements for Anime Background Remover")

if not launch.is_installed("onnxruntime-gpu"):
    launch.run_pip("install onnxruntime-gpu", "requirements for Anime Background Remover")

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirements for Anime Background Remover")

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", "requirements for Anime Background Remover")