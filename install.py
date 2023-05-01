import launch

for dep in ['onnx', 'onnxruntime', 'numpy']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for ABG_extension")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "opencv-python")

if not launch.is_installed("PIL"):
    launch.run_pip("install Pillow", "Pillow")
