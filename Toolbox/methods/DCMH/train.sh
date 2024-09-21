#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python demo_DCMH.py --dataname 'flickr' --bits 32
