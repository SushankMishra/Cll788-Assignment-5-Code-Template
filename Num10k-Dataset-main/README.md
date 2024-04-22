# Num10k Dataset

This repository contains the `Num10k` Dataset used for Assignment 5 of the course CLL788 at IIT Delhi. It is a synthetically created dataset, specifically designed for Optical Character Recognition (OCR) tasks, and comprises solely of numerical digits.

## Dataset Structure

The `Num10k` consists of 11,000 .jpg images divided into two subsets:

- **Training Set**: This subset contains 10,000 images.
- **Validation Set**: This subset consists of 1,000 images.

Each image represents a multi-digit number consisting of digits 0-9. The dataset is divided into two subfolders: `train` and `val`. Each folder contains a `labels.txt` file and the corresponding .jpg images. The `labels.txt` file maps each image to its ground truth, separated by a tab character.

Here's how the dataset looks like:

```
Num10kDataset
  - train
    - labels.txt
    - 0.jpg
    - 1.jpg
    - 2.jpg
    - 3.jpg
  - val
    - labels.txt
    - 0.jpg
    - 1.jpg
    - 2.jpg
    - 3.jpg
```

## Purpose
The purpose of this dataset is to provide a OCR task for the students to implement a basic OCR model. Currently, the characters are just digits 0-9, but it can be extended to include alphabets and other characters as well.

## Course Details

- **Course**: CLL788, IIT Delhi
- **Course Coordinator**: Dr. Manojkumar Ramteke
- **Teaching Assistant**: Abdur Rahman