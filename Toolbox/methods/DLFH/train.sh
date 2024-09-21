#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_DLFH('64', 'flickr'); quit;" 
cd ..
