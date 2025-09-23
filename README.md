# SMNV3-GRU
SMNV3-GRU: Sequential MobileNetV3 with Gate-Recurrent-Unit Model for Continuous Human Action Recognition and its Application

<h3 align="center">Model Architecture</h3>

<p align="center">
<img width="503" height="447" alt="image" src="https://github.com/user-attachments/assets/1cd7e7ec-6193-46ab-ad8e-be4168e2b198" />

</p>

---

## 📘 Overview
The proposed SMNV3-GRU sequentially integrates a small version of MobileNetV3-based feature extractor with GRU-based temporal modeling, enabling efficient recognition of continuous human motion sequences.

It supports:

- **Model training and testing**  
- **Real-time inference on a PC**
## 📊 Dataset
<p align="center">
<img width="503" height="125" alt="image" src="https://github.com/user-attachments/assets/7695f448-0fd9-44db-8380-5511f6715c39" />
</p>
<p align="center"><i> Six designed wireless gestures.</i></p>
###  Sequential Images for Six Designed Human Actions  
 
Each sequence contains color images with a resolution of **224×224**, prepared for training and evaluation purposes.  

📂 [Link](https://drive.google.com/drive/folders/1p6aKKVxHDFl7NWXeQu0tIyUyt5PiVJxl?usp=drive_link)
## 🚀 Performance

### Training and Validation Responses
<table align="center">
  <tr>
    <td align="center">
      <img width="320" height="240" src="https://github.com/user-attachments/assets/0ab5d939-3f4c-4f33-b375-f187205c062d" /><br>(a)
    </td>
    <td align="center">
      <img width="320" height="240" src="https://github.com/user-attachments/assets/742af9a7-8caf-493c-953e-7603ae041daa" /><br>(b)
    </td>
    <td align="center">
      <img width="320" height="240" src="https://github.com/user-attachments/assets/54bcd825-2836-4473-98c2-113ba5a7c0c9" /><br>(c)
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="320" height="240" src="https://github.com/user-attachments/assets/817c1f44-784f-414d-9e76-593ef65cede9" /><br>(d)
    </td>
    <td align="center">
      <img width="320" height="240" src="https://github.com/user-attachments/assets/afc6dafa-1a46-4de9-b393-90b30a092323" /><br>(e)
    </td>
    <td align="center">
      <img width="320" height="240" src="https://github.com/user-attachments/assets/2cdd8f76-ae35-4a11-8fc7-09e16d291eb4" /><br>(f)
    </td>
  </tr>
</table>
Confusion matrix

<p align="center">
<img width="677" height="536" alt="image" src="https://github.com/user-attachments/assets/b1e88cf3-8068-473c-a7b5-93513a8f600a" />


## 🖼️ Application to human-UAV interactions
### Indoor quadrotor 
<p align="center">
  <img width="503" height="167" alt="image" src="https://github.com/user-attachments/assets/ab33b6ec-d89a-40b6-af7c-8e8a808a3ac1" />

</p>

<p align="center"><i> Distributed UWB Network and UAV control setup</i></p>

### Experiment 1  
Extra humans inside of FOV [Video](https://youtu.be/JeYRMwli88Q?si=l2oAThJ-h6hs4ItM)
<p align="center">
  <img width="255" height="210" alt="loss-curve" src="https://github.com/user-attachments/assets/ea88ade5-4119-4a5e-bd08-c23e29214533" />
  <img width="244" height="208" alt="confusion-matrix" src="https://github.com/user-attachments/assets/0576a573-6782-481b-ab7c-dfda05055111" />
</p>

<p align="center">Experiment 1 results </i></p>

### Experiment 2  
Dark illumination with extra humans inside of FOV [Video](https://youtu.be/VHAf1cZUyL8?si=jcO0CYTae0_jyE38)
<p align="center">
<img width="250" height="201" alt="image" src="https://github.com/user-attachments/assets/34164b6a-e5e3-4f98-b640-497f1dc0c812" />
<img width="251" height="203" alt="image" src="https://github.com/user-attachments/assets/3f9c794a-af2b-43f5-93d9-ce579f955ead" />
</p>

<p align="center">Experiment 2 results </i></p>

### Experiment 3  
Different operated humans with dark illumination and extra humans inside of FOV [Video](https://youtu.be/S7VNDHIUlhk?si=bSFozpxuWWEenp5P)
<p align="center">
<img width="252" height="183" alt="image" src="https://github.com/user-attachments/assets/d6b5da3e-776e-4e04-adde-1208c0f481ba" />
<img width="245" height="182" alt="image" src="https://github.com/user-attachments/assets/ed501984-8289-420d-b230-464b1b080b32" />

<p align="center">Experiment 2 results </i></p>

