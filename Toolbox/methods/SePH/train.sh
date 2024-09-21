#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_SePH('64', 'flickr'); quit;" 
cd ..
