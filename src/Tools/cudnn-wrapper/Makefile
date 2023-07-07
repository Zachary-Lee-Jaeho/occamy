all:
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcudnn -c CUDNNConvFunc.cpp -o CUDNNConvFunc.o
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcudnn -c wrapCUDNNMalloc.cpp -o wrapCUDNNMalloc.o
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcudnn -c wrapCUDNNFree.cpp -o wrapCUDNNFree.o
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcudnn -c wrapCUDNNMemcpy.cpp -o wrapCUDNNMemcpy.o

cl:
	rm -rf CUDNNConvFunc.o CUDNNConvFunc
	rm -rf wrapCUDNNMalloc.o wrapCUDNNMalloc
	rm -rf wrapCUDNNFree.o wrapCUDNNFree
	rm -rf wrapCUDNNMemcpy.o wrapCUDNNMemcpy

obj:
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcudnn -c CUDNNConvFunc.cpp -o CUDNNConvFunc.o
	objdump -t CUDNNConvFunc.o
