#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_IMH('64', 'flickr'); quit;" 
cd ..
