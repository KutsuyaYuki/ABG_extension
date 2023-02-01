import launch

if not launch.is_installed("onnx"):
    launch.run_pip("install onnx")
    print("Installing onnx...")

if not launch.is_installed("onnxruntime-gpu"):
    launch.run_pip("install onnxruntime-gpu")
    print("Installing onnxruntime-gpu...")

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python")
    print("Installing opencv-python...")

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy")
    print("Installing numpy...")

if not launch.is_installed("Pillow"):
    launch.run_pip("install Pillow")
    print("Installing Pillow...")
