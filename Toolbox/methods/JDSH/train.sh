#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python demo_JDSH.py --dataname 'flickr' --bits 32
