#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python demo_DJSRH.py --dataname 'flickr' --bits 32

