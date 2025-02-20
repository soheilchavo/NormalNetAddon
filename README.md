## 🔍 Description
The NormalNet addon is a companion repositry to NormalNet, a project for creating generative models to create PBR materials from diffuse textures.
The addon has models for AO, Normal, Metallic, and Roughness maps, simply select your diffuse texture in the shader editor and the addon will make
an AI-generated PBR for you. The addon is still quite janky, so use with caution, and feel free to create a PR and change anything you want.

## 📚 Dependencies
In order to run the addon, you must first install the following dependencies for Blender using the terminal:

```console
cd <Your Blender Directory>/Contents/Resources/3.x/python/bin
python3.x -m pip install opencv-python
python3.x -m pip install torch
```

## 🛠️ Usage

Once the dependencies are correctly installed, download the zip file for this repo and install it in Blender.

To use the addon, open the shader editor and press Shift+A or go to add in the menu to create a new node.

Select 'PBR Generator', select your diffuse image, and press 'Generate'.
<img width="618" alt="image" src="https://github.com/user-attachments/assets/7711ba54-005e-41e2-8490-cc9f42268d40" />

And Volia! You''re all set!
<img width="793" alt="image" src="https://github.com/user-attachments/assets/e65d2c27-5dfe-4cef-8a29-c60f160c0efc" />


## 🚧 Feature Roadmap

- ✨ Less Janky UI
- ✨ Better Models
