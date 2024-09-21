#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_LSSH('16', 'flickr'); quit;" 
cd ..
