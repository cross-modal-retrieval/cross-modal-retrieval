#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_DCH('64', 'flickr'); quit;" 
cd ..
