# DeepWaterFraction: Precision Estimation of Sub Pixel Scale Water Areas in Landsat Imagery

## Description

This repository contains code and pre-trained models for a self-training deep learning methodology that estimates surface water areas at a sub-pixel scale, as described in our research paper "[Paper Title]". The presented method provides significantly improved accuracy in detecting and estimating water bodies at fine scales, demonstrated by a pixel-wise root mean squared error for surface water area fraction of 14.8%. 

Specifically, our method reduces error rates by 53.4% for water bodies with a minimum area of 0.001 km² and by 27.4% for those with a minimum area of 0.01 km². Additionally, when applied to streamflow gauges, the method significantly improved the correlation against observed streamflow. The repository also provides scripts for visualizing the results of our model.

Our research opens a new avenue for accurate space-based estimation of river discharge, making it a crucial asset for global water resource monitoring.

## Content of the Repository

- **main.py**: This is the main script which loads the data, applies the deep learning model, visualizes the results, and saves them.
- **dwf.py**: This script contains the implementation of the Deep Water Fraction (DWF) estimation model, including a custom Gated Convolution 2D Activation module.
- **utilities.py**: [Please fill out a brief description of this file here.]
- **Pre-trained weights**: These are the weights of the models that have been pre-trained on our dataset. They can be found in the directory [specify directory].
- **Data**: This repository does not host the data used in the research due to its size. However, instructions for acquiring the necessary data are detailed in the README file.
- **README**: The README file contains detailed information on how to setup and run the provided scripts, as well as information on how to download and prepare the required data.

## Usage 

To run the code, simply download the repository and follow the instructions in the README file. 

## Future Work

This repository will be actively maintained and updated as we refine our model and expand our research. We welcome suggestions from the community. 

## Contact Information

If you encounter any issues or have any questions about the code or the research, please feel free to contact us at [your email].

## Citation

If you find our work helpful in your research, please consider citing our paper:

```
[Your Bibtex citation]
```

## Acknowledgements

This research was supported by [insert your acknowledgements here]. We thank all our team members and contributors for their hard work and valuable insights. 

## License

This project is licensed under the [insert your license here] License - see the LICENSE.md file for details.
