import launch

for dep in ['onnx', 'onnxruntime', 'opencv-python', 'numpy', 'Pillow']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for ABG_extension")