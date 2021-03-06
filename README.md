# Real Image Manipulation under Domain-Guided-Noise-Optimization Mechanism

Do facial image Manipulation on Google Colab: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iHjxBLK1H2N4FKTamYvkmIiPjMGw-_YL?authuser=1#scrollTo=AjyF24k-UXyb)

Notice: the image is display properly now. The .gif file is converted by using website tool and some details are missing.

Image2Latent Semantic Results
------------
</div>
<img src="./DNI-Code/result_file/semantic.png" width="512" height="512"> 
</div>


## Encoding an image into StyleGAN2 latent space

</div>
<img src="./DNI-Code/result_file/encoding.png" width="512" height="256"> 
</div>

Manipulate attribute Results
------------
| Modify Faces | | |
| :-- | :-- |:-- |
| Eye Closed | Pose | Star|
|</div><img src="./DNI-Code/result_file/eye.gif" width="256" height="256"> </div>|</div><img src="./DNI-Code/result_file/pose.gif" width="256" height="256"></div>|</div><img src="./DNI-Code/result_file/star.gif" width="256" height="256"></div>|

Introduction Notebook
------------------

We provide a jupyter notebook example to show how to use DNI for facial image manipulation: `/DNI-code/DNI.ipynb`.

We also provide a colab version of the notebook: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iHjxBLK1H2N4FKTamYvkmIiPjMGw-_YL?authuser=1#scrollTo=AjyF24k-UXyb). Be sure to select the GPU as the accelerator in runtime options.

Psp Encoder architecture
----------------------
More details for the Psp model, please see [here](https://github.com/eladrich/pixel2style2pixel).

The graphical of Psp model. The whole architecture is in here: `/DNI-code/results/Psp_model.pdf`.

</div>
<img src="./DNI-Code/result_file/PSp.jpg" width="512" height="256"> 
</div>

## Youtube Video

Please visit the [Youtube](https://www.youtube.com/watch?v=KrcCRZs7J98&feature=youtu.be) video for better understanding!

## Failure cases
There are some failure cases contained in our method when editing a specific attribute, which is based on cosine similarity metric. 
</div>
<img src="./DNI-Code/result_file/Failure.jpg" width="512" height="128">
<img src="./DNI-Code/result_file/case.jpg" width="512" height="128">
</div>


## Related Projects

**[StyleFlow](https://github.com/RameenAbdal/StyleFlow) | [PsP Encoder](https://github.com/eladrich/pixel2style2pixel) | [StyleGAN2Encoder](https://github.com/bryandlee/stylegan2-encoder-pytorch) | [StyleGAN2](https://github.com/NVlabs/stylegan2)**
