
include ../../../../common/common.mk

BENCHMARK=mri-fhd
VERSION=klee
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/file.o obj/ocl-wrapper.o obj/computeFH-cpu.o obj/computeFH-gpu.o
	$(CXXLD) -lOpenCL $^ -o ../../bin/$(BENCHMARK)-$(VERSION)

obj/main.o: main.cpp ../base/computeFH.h ../opencl/computeFH.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c $< -o $@

obj/computeFH-cpu.o: ../base/computeFH.cc ../base/computeFH.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -c $< -o $@

obj/computeFH-gpu.o: ../opencl/computeFH.cpp ../opencl/computeFH.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -D DISABLE_TIMERS -c $< -o $@

obj/file.o: ../opencl/file.cpp ../opencl/file.h
	$(CXX) $(CXXFLAGS) -c ../opencl/file.cpp -o obj/file.o

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

