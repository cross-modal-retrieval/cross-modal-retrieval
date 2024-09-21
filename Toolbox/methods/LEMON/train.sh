#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_LEMON('64', 'flickr'); quit;" 
cd ..
