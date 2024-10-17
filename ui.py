import gradio as gr
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torchvision import transforms as tfs
import torch
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms

def load_net(net, checkpoints_dir, net_name, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, net_name)
    save_path = os.path.join(checkpoints_dir, save_filename)
    net.load_state_dict(torch.load(save_path))
    print('load_net{}: {}'.format(net_name, save_filename))

class DeblurGenerator(nn.Module):
    def __init__(self, padding_type='reflect'):
        super(DeblurGenerator, self).__init__()
        # conv-->(downsamping x 2)-->(resnblock x 9)-->(deconv x 2)-->conv-->
        deblur_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True)
        ]

        deblur_model += [
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=True),
            nn.ReLU(True)
        ]

        for i in range(9):
            deblur_model += [
                Resblock(256, padding_type)
            ]

        deblur_model += [
            nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True),
        ]

        deblur_model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*deblur_model)

    def forward(self, x):
        res = x
        out = self.model(x)
        return torch.clamp(out + res, min=-1, max=1)


class Resblock(nn.Module):
    def __init__(self, channel, padding_type):
        super(Resblock, self).__init__()
        self.conv_block = self.build_conv_block(channel, padding_type)

    def build_conv_block(self, channel, padding_type):
        conv_block = []

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [nn.ZeroPad2d(1)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(channel),
                       nn.ReLU(True)]

        conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        conv_block = self.conv_block(x)
        return conv_block + x


# Instantiate your model
model = DeblurGenerator()
load_net(model, 'D:\Downloads NA C\MP Project\ML Proej', 'G', 1)
# if torch.cuda.is_available():
#     model = model.cuda(0)

# # Preprocess function for the input image
# def preprocess(image):
#     image = image.resize((360, 360), Image.BICUBIC)
#     print(type(image))
#     transform = tfs.Compose([tfs.ToTensor(),
#                           tfs.Normalize((0.5, 0.5, 0.5),
#                                                (0.5, 0.5, 0.5))])
#     return Variable(transform(image))

# # Post-process function to rescale the output and convert it to image format
# def postprocess(output_tensor, width, height):
#     image = image_recovery(output_tensor)
#     print("image:", image.shape)
#     resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
#     return resized_image

# # Function to process the image through the model
# def generate_image(image):
#     width, height = image.size
#     print(image.size)
#     preprocessed_image = preprocess(image)
#     print("preprocessed:", preprocessed_image.shape)
#     output_image_tensor = model(preprocessed_image)
#     print("output_image", output_image_tensor.shape) 
#     result_image = postprocess(output_image_tensor, width, height)
#     return result_image

# def image_recovery(image_tensor, imtype=np.uint8):
#     image_numpy = image_tensor.cpu().float().detach().numpy()
#     print(image_numpy.shape)
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     return image_numpy.astype(imtype)

# # Create a Gradio Interface
# # interface = gr.Interface(
# #     fn=generate_image,  # Function that processes the image
# #     inputs=gr.Image(type="pil"),  # Input: Image file
# #     outputs=gr.Image(type="pil"),  # Output: Image file
# #     title="Image Generator",
# #     description="Upload an image and pass it through a generator model to get a new image"
# # )

# interface = gr.Interface(
#     fn=generate_image,  # Function that processes the image
#     inputs=gr.Image(type="pil"),  # Input: Image file (unblurred)
#     outputs=[gr.Image(type="pil", label="Blurred Image"), gr.Image(type="pil", label="Deblurred Image")],  # Two outputs: blurred and deblurred
#     title="Image Blurring and Deblurring",
#     description="Upload an image, the input will be blurred and passed through the model to generate a deblurred output"
# )
# # Launch the interface
# interface.launch()


import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as tfs

# Assuming your DeblurGenerator class is already defined above, with the model loaded

# Preprocess function for the input image
def preprocess(image):
    image = image.resize((360, 360), Image.BICUBIC)  # Resize the image to 360x360
    image_np = np.array(image)  # Convert to NumPy array
    return image_np

# Function to apply Gaussian blur to the image
def blur_image(image):
    blurred_image = cv2.GaussianBlur(image, (11, 11), 1.6)  # Apply Gaussian blur
    return blurred_image

# Post-process function to convert back to PIL image after processing
def postprocess(image, width, height):
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(resized_image)  # Convert NumPy array back to PIL image

# Convert the blurred image to a format for the model (tensor)
def prepare_for_model(image):
    transform = tfs.Compose([tfs.ToTensor(),
                             tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_tensor = transform(Image.fromarray(image))
    return Variable(image_tensor).unsqueeze(0)  # Add batch dimension

# Function to recover the deblurred image back from tensor to image
def image_recovery(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy[0], (1, 2, 0)) + 1) / 2.0 * 255.0  # Remove batch dimension
    return image_numpy.astype(imtype)

# Main function to blur the input image and then deblur using the model
def generate_image(image):
    width, height = image.size
    
    # Step 1: Preprocess the input image (resize to 360x360)
    preprocessed_image = preprocess(image)
    
    # Step 2: Blur the image and return the blurred image (output1)
    blurred_image = blur_image(preprocessed_image)
    
    # Step 3: Prepare the blurred image for the model (convert to tensor)
    image_for_model = prepare_for_model(blurred_image)
    
    # Step 4: Pass the blurred image through the deblurring model
    deblurred_image_tensor = model(image_for_model)
    
    # Step 5: Recover the deblurred image back to image format
    deblurred_image = image_recovery(deblurred_image_tensor)
    
    # Resize both blurred and deblurred images back to original dimensions
    output1 = postprocess(blurred_image, width, height)  # Blurred image
    output2 = postprocess(deblurred_image, width, height)  # Deblurred image
    
    return output1, output2  # Return both blurred and deblurred images

# Create a Gradio Interface with two outputs
interface = gr.Interface(
    fn=generate_image,  # Function that processes the image
    inputs=gr.Image(type="pil"),  # Input: Image file (unblurred)
    outputs=[gr.Image(type="pil", label="Blurred Image"), gr.Image(type="pil", label="Deblurred Image")],  # Two outputs: blurred and deblurred
    title="Image Blurring and Deblurring",
    description="Upload an image, the input will be blurred and passed through the model to generate a deblurred output"
)

# Launch the interface
interface.launch()