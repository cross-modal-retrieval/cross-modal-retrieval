#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python demo_SSAH.py --dataname 'flickr' --bits 32
