
include ../../../../common/common.mk

BENCHMARK=rpes
VERSION=opencl
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/common.o obj/ocl-wrapper.o
	$(CXXLD) -lOpenCL obj/common.o obj/ocl-wrapper.o obj/main.o -o ../../bin/$(BENCHMARK)-$(VERSION)
	cp crys_kernel.cl ../../bin/

obj/main.o: main.cpp shell.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c main.cpp -o obj/main.o

obj/common.o: $(COMMON_DIR)/common.cpp $(COMMON_DIR)/common.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/common.cpp -o obj/common.o

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

