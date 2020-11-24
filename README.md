# Mobilenet_gender_classification
* paper reference: Bengali Ethnicity Recognition and Gender Classification Using CNN & Transfer Learning

# Implementation
* Ubuntu 18.04
* Python 3.7.8
* tensorflow 2.3.0

# Detail
* This version is mobilenet training in this paper.
* 2-fold cross validiation
* Use test accuracy during training instead of validation accuracy
* For my generated images (AFAD) from proposed method, first fold acc was 81.71 % (train-16,081, test-16,081)
* Second fold acc 81.46 % (AFAD, train-16,081, test-16,081)
* First fold acc 92.81% (Morph)
* Second fild acc 93.74 % (Morph)
* Final - 81.59 % for AFAD, 93.28 % for Morph

* First fold acc 97.93 % (original AFAD)
* Second fold acc 99.79 % (original AFAD)
* Orginal acc 98.86 % (original AFAD)

* First fold acc 99.56 % (original Morph)
* Second fold acc 99.95 % (original Morph)
* Orginal acc 99.75 % (original Morph)
* Train (original AFAD), Test (generated AFAD) 1-fold acc is 98.3 %
