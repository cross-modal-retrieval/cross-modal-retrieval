#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_CMFH('64', 'flickr'); quit;" 
cd ..
