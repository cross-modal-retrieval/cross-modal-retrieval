#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_CVH('64', 'flickr'); quit;" 
cd ..
