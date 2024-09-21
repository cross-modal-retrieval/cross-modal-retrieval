#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_SCM('64', 'flickr'); quit;" 
cd ..
