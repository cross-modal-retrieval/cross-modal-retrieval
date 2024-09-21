#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_CCQ('64', 'flickr'); quit;" 
cd ..
