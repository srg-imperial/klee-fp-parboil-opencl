
include ../../../../common/common.mk

BENCHMARK=mri-q
VERSION=opencl
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/file.o obj/common.o obj/ocl-wrapper.o obj/computeMRIQ.o
	$(CXXLD) -lOpenCL $^ -o ../../bin/$(BENCHMARK)-$(VERSION)
	cp computeQ.cl ../../bin/

obj/main.o: main.cpp computeMRIQ.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c main.cpp -o obj/main.o

obj/file.o: file.cpp file.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -c file.cpp -o obj/file.o

obj/computeMRIQ.o: computeMRIQ.cpp computeMRIQ.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c computeMRIQ.cpp -o obj/computeMRIQ.o

obj/common.o: $(COMMON_DIR)/common.cpp $(COMMON_DIR)/common.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/common.cpp -o obj/common.o

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

