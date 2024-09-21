#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_BATCH('64', 'flickr'); quit;" 
cd ..
