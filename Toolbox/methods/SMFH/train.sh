#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_SMFH('64', 'flickr'); quit;" 
cd ..
