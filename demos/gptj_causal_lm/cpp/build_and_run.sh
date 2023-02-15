#g++ main.cpp -L./ -lbingfirtinydll -o main && LD_LIBRARY_PATH=./ ./main
g++ main.cpp -L./ -l:libblingfiretokdll_static.a -l:libfsaClient.a -o main && LD_LIBRARY_PATH=./ ./main
