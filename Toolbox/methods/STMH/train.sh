#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_STMH('64', 'flickr'); quit;" 
cd ..
