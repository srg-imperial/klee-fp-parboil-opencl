
include ../../../../common/common.mk

BENCHMARK=cp
VERSION=klee
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/ocl-wrapper.o obj/cenergy.o obj/cuenergy.o
	$(CXXLD) -lOpenCL $^ -o ../../bin/$(BENCHMARK)-$(VERSION)

obj/main.o: main.cpp ../base/cenergy.h ../opencl/cuenergy.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c $< -o $@

obj/cenergy.o: ../base/cenergy.c ../base/cenergy.h
	$(CC) $(CFLAGS) -I$(COMMON_DIR) -c $< -o $@

obj/cuenergy.o: ../opencl/cuenergy.cpp ../opencl/cuenergy.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -D DISABLE_TIMERS -c $< -o $@

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

