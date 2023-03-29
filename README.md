<p align="center">
 # ABG extension
</p>

<p align="center">
	<a href="https://github.com/KutsuyaYuki/ABG_extension/stargazers"><img src="https://img.shields.io/github/stars/KutsuyaYuki/ABG_extension?style=for-the-badge"></a>
	<a href="https://github.com/KutsuyaYuki/ABG_extension/issues"><img src="https://img.shields.io/github/issues/KutsuyaYuki/ABG_extension?style=for-the-badge"></a>
	<a href="https://github.com/KutsuyaYuki/ABG_extension/contributors"><img src="https://img.shields.io/github/last-commit/KutsuyaYuki/ABG_extension?style=for-the-badge"></a>
</p>

## Installation

 1. Install extension by going to Extensions tab -> Install from URL -> Paste github URL and click Install.
 2. After it's installed, go back to the Installed tab in Extensions and press Apply and restart UI.
 3. Installation finished.

## Usage

### txt2img

 1. In the bottom of the WebUI in Script, select **ABG Remover**.
 2. Select the desired options: **Only save background free pictures** or **Do not auto save**.
 3. Generate an image and you will see the result in the output area.

### img2img

 1. In the bottom of the WebUI in Script, select **ABG Remover**.
 2. Select the desired options: **Only save background free pictures** or **Do not auto save**.
 3. **IMPORTANT**: Set **Denoising strength** to a low value, like **0.01**

Based on https://huggingface.co/spaces/skytnt/anime-remove-background
