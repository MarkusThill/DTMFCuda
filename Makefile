IDIR=..
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lcudart -lcuda --std c++17

.PHONY: clean build run

build:	*.cu
	mkdir -p build && cd build/ && $(COMPILER) $(COMPILER_FLAGS) ../*.cu -o dtmfCuda

clean:
	rm -rf build/

run:
	cd build;\
	./dtmfCuda $(ARGS)

all: clean build run
