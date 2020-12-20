import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from skimage import io, transform
import base64
import requests
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pickle
import matplotlib

class Model(nn.Module):

	def __init__(self, n):

		super(Model, self).__init__()

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

		self.conv3_32 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.conv32_32 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv64_32 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.conv64_3n = nn.Conv2d(64, 3*n, kernel_size=3, stride=1, padding=1)

	# Total of 7 layers with skip connections
	def forward(self, inpt):

		output1 = self.relu(self.conv3_32(inpt.float()))
		output2 = self.relu(self.conv32_32(output1))
		output3 = self.relu(self.conv32_32(output2))
		output4 = self.relu(self.conv32_32(output3))

		output5 = self.relu(self.conv64_32(torch.cat([output4, output3], dim=1)))
		output6 = self.relu(self.conv64_32(torch.cat([output5, output2], dim=1)))

		output7 = self.tanh(self.conv64_3n(torch.cat([output6, output1], dim=1)))

		return output7

def Rescale(sample, output_h, output_w):
    image = sample
    
    new_h, new_w = output_h, output_w

    img = transform.resize(image, (new_h, new_w))
  
    return img

def ToTensor(sample):
    image = sample
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)

def Enhance(img_array):
    h,w,c = img_array.shape
    res = 512
    if c != 3:
        st.warning('The input image is not of type RGB, try using another image.')
        st.stop()
    if h != w:
        st.warning('The input image ratio is not 1:1. Hence, you may observe distortions in the result.')
    img = Rescale(img_array, res,res)
    img = ToTensor(img)
    n = 8
    model_path = 'model2.pkl'
    model = pickle.load(open(model_path, 'rb'))
    img = img.unsqueeze(0)
    a = model(img)
    a_n = a.reshape(1, n, 3, res, res)
    LE = img
    for iter in range(n):
	    LE = LE + torch.mul(torch.mul(a_n[0][iter], LE), (torch.ones(LE.shape) - LE))
    np_array = LE.squeeze().cpu().detach().permute(1,2,0).numpy()
    np_array = np_array*255
    np_array = np_array.astype('uint8')
    LE = Image.fromarray(np_array,'RGB')
    return LE    

st.set_page_config(
    page_title="Light It Up!",
    page_icon=":first_quarter_moon:",
)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title(":first_quarter_moon: Light It Up!")
st.subheader("Welcome to Low-Light Image Enhancer")

def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a download="output.jpg" href="data:file/jpg;base64,{img_str}">Download Enhanced Image</a>'
    return href

image_data = st.file_uploader("Upload Square Image :", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

st.markdown("<p style='text-align: center;'>OR</p>",unsafe_allow_html=True)

image_url = st.text_input("Image URL : ")

if image_data is None and image_url:
    try:
        response = requests.get(image_url)
        image_data = BytesIO(response.content)
    except:
        st.write("Please enter a valid URL")

            
if st.button('Enhance'):
    if image_data is not None:        
        my_bar = st.progress(0)
        for percent_complete in range(51):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 50)
            
        input_image = Image.open(image_data)
        
        img_array = np.array(input_image)
        
        output_image = Enhance(img_array)
           
        col1, col2 = st.beta_columns(2)
        with col1:
            st.image(input_image, use_column_width=True)

        with col2:
            st.image(output_image, use_column_width=True)    
    
    else:
        st.warning('Please Upload an Image!')
        st.stop()

    st.markdown(get_image_download_link(output_image), unsafe_allow_html=True)
    
    st.title("Interested in knowing how the model works?")
    st.write("Well, we have trained a lightweight Deep Curve Estimation Network (DCE-Net) to estimate pixel-wise and high-order curves for dynamic range adjustment of a given image. This is achieved through a set of carefully formulated non-reference loss functions, which implicitly measure the enhancement quality and drive the learning of the network. Our method is efficient as image enhancement can be achieved by an intuitive and simple nonlinear curve mapping. Despite its simplicity, we show that it generalizes well to diverse lighting conditions. In contrast to other deep learning-based methods, our model implements the Zero-Reference method in which it is trained without any reference images i.e. it does not require any paired or unpaired data during the training process as in existing CNN-based and GAN-based methods.")
    
    with st.echo():
        def spatial_loss(i, o):
            i = avg_pool_4(i)
            o = avg_pool_4(o)
            d_i = nn.functional.conv2d(i, weights_spatial, padding=1, stride=1)
            d_o = nn.functional.conv2d(o, weights_spatial, padding=1, stride=1)
            d = torch.square(torch.abs(d_o) - torch.abs(d_i))
            s = torch.sum(d,dim=1)
            l_spa = torch.mean(s)
            return l_spa
    with st.echo():
        def exposure_loss(o):
            E = 0.6
            o = avg_pool_16(o)
            o = torch.abs(o - E*torch.ones(o.shape))
            l_exp = torch.mean(o)
            return l_exp
    with st.echo():
        def color_loss(o):
            avg_intensity_channel = torch.mean(o, dim=(2,3))
            avg_intensity_channel_rolled = torch.roll(avg_intensity_channel, 1, 1)
            d_j = torch.square(torch.abs(avg_intensity_channel - avg_intensity_channel_rolled))
            l_col = torch.mean(torch.sum(d_j, dim=1))
            return l_col
    with st.echo():
        def illumination_loss(A, size, batch_size, n):
            h_grad = nn.functional.conv2d(A, w_h, padding=1, groups=n*3)
            v_grad = nn.functional.conv2d(A, w_v, padding=1, groups=n*3)
            h_grad = h_grad.reshape(batch_size, n, 3, size, size)
            v_grad = v_grad.reshape(batch_size, n, 3, size, size)
            h_grad = torch.mean(h_grad, dim=(3,4))
            v_grad = torch.mean(v_grad, dim=(3,4))
            grad = torch.square(torch.abs(h_grad) + torch.abs(v_grad))
            grad = torch.sum(grad, dim=2)
            l_tva = torch.mean(grad)
            return l_tva
    with st.echo():
        def compute_losses(input, output, A, size, batch_size, n):
            w_col = 0.5
            w_tva = 20
            l_spa = spatial_loss(input, output)
            l_exp = exposure_loss(output)
            l_col = color_loss(output)
            l_tva = illumination_loss(A, size, batch_size, n)
            return l_spa + l_exp + w_col*l_col + w_tva*l_tva
    st.subheader("Team")
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        ankur = Image.open("ankur.jpeg")
        st.image(ankur, use_column_width=True)
        st.write("""<a style='display: block; text-align: center; text-decoration: none' href="https://www.linkedin.com/in/ankur-chemburkar-01020a180/" target="_blank">Ankur</a>""",unsafe_allow_html=True,)

    with col2:
        devang = Image.open("devang.jpg")
        st.image(devang, use_column_width=True)
        st.write("""<a style='display: block; text-align: center; text-decoration: none' href="https://www.linkedin.com/in/djrobin17/" target="_blank">Devang</a>""",unsafe_allow_html=True,)
    with col3:
        hashir = Image.open("hashir.png")
        st.image(hashir, use_column_width=True)
        st.write("""<a style='display: block; text-align: center; text-decoration: none' href="https://www.linkedin.com/in/hashirkk/" target="_blank">Hashir</a>""",unsafe_allow_html=True,)

    
    st.write("If you liked our work, you can support us here :)")
    components.html("""<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="djrobin17" data-color="#fafafa" data-emoji=""  data-font="Poppins" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#FFDD00" ></script>""")
    st.balloons()   
# feedback = st.text_input('Any Feedback?')
# btn = st.button('Send')
# if btn:
#     if feedback:
#         with open("feedback.txt", "a") as f:
#             f.write(feedback)
#             f.close()
#         st.balloons()



