root_dir=`pwd`
cat example-paths.yaml | sed  "s/\/workspace\/project/${root_dir//\//\\/}/g" > paths.yaml
