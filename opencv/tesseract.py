#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:34:42 2018

@author: dirac
"""

import pytesseract
from PIL import Image

img=Image.open('data/auth1.jpg')
a=pytesseract.image_to_string(img)
print (a)