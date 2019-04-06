# Twining digits

### 4. Generate video
```
python write_numbers.py
```
this takes a sequence of digits as inputs and generates an output where the digits twine from one to the other sequentially.
**Though to run this file one has to download the [model-4000.h5](https://github.com/aniruddhas435/ACGAN/blob/master/model-4000.h5) file**

---

## Introduction

This application turns a _sequence of digits_ into a video where their handwritten representation _twine from one to the other_. Each frame of this video comes from an ACGAN generator function.

## Description

The above task is carried out by an implementation of an ACGAN(_Auxiliary Classifier Generative Adverserial NetWorks_). Except for the fact that here we can _**ambiguous**_ _class intent_ to the generator. This is acheived through a representtion of the _class intent_ as a vector with the respective probabilities for each class. A _**custom embedding**_ has been created to learn the representation of this kind which enables us to build the model. The results that come out are quite convincing. One can download the [writing_numbers.avi](https://github.com/aniruddhas435/ACGAN/blob/master/twining_digits.avi) video to see the performance for himself.
