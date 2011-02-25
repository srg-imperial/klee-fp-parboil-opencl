
include ../../../../common/common.mk

BENCHMARK=mri-q
VERSION=klee
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/file.o obj/ocl-wrapper.o obj/computeMRIQ.o obj/computeQ.o
	$(CXXLD) -lOpenCL $^ -o ../../bin/$(BENCHMARK)-$(VERSION)

obj/main.o: main.c ../base/computeQ.h ../opencl/computeMRIQ.h
	$(CC) $(CFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c main.c -o obj/main.o

obj/computeQ.o: ../base/computeQ.cc ../base/computeQ.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -c ../base/computeQ.cc -o obj/computeQ.o

obj/computeMRIQ.o: ../opencl/computeMRIQ.cpp ../opencl/computeMRIQ.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c ../opencl/computeMRIQ.cpp -o obj/computeMRIQ.o

obj/file.o: ../opencl/file.cpp ../opencl/file.h
	$(CXX) $(CXXFLAGS) -c ../opencl/file.cpp -o obj/file.o

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

