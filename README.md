# DeepWaterFraction: Precision Estimation of Sub Pixel Scale Water Areas in Landsat Imagery

## Description

This repository contains code and pre-trained models for DeepWaterFraction, a self-training deep learning methodology that estimates surface water areas at a sub-pixel scale. The presented method provides significantly improved accuracy in detecting and estimating water bodies at fine scales, demonstrated by a pixel-wise root mean squared error for surface water area fraction of 14.8%. 

Specifically, our method reduces error rates by 52.7% for water bodies with a minimum area of 0.001 km² and by 22.3% for those with a minimum area of 0.01 km². Additionally, when applied to streamflow gauges and water level stations, the method significantly improved the correlation against observed streamflow and water level. The repository provides scripts for inferencing and visualizing the results of our model.

The pre-trained models and sample data associated with this research are hosted on Zenodo (links provided below). 

![Comparison Figure](./data/image1.png)
![Comparison Figure](./data/image2.png)

## Content of the Repository

- **main.py**: This is the main script which loads the data, applies the deep learning model, visualizes the results, and saves them.
- **dwf.py**: This script contains the implementation of the Deep Water Fraction (DWF) estimation model.
- **Pre-trained weights**: These are the weights of the models that have been pre-trained on our dataset. They can be found in the directory [specify directory].

## Usage 

To run the code, simply download the repository and run the **main.py** file. 

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
