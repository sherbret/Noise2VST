#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request

url = "https://github.com/cszn/KAIR/releases/download/v1.0/"
for model in ["ffdnet_gray.pth", "ffdnet_color.pth", "drunet_gray.pth", "drunet_color.pth"]:
    urllib.request.urlretrieve(url + model, "./pretrained_weights/" + model)