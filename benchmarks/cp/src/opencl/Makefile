
include ../../../../common/common.mk

BENCHMARK=cp
VERSION=opencl
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/cuenergy.o obj/common.o obj/ocl-wrapper.o
	$(CXXLD) -lOpenCL $^ -o ../../bin/$(BENCHMARK)-$(VERSION)
	cp cuenergy_pre8_coalesce.cl ../../bin/

obj/main.o: main.cpp cuenergy.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c main.cpp -o obj/main.o

obj/cuenergy.o: cuenergy.cpp cuenergy.h
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c cuenergy.cpp -o obj/cuenergy.o

obj/common.o: $(COMMON_DIR)/common.cpp $(COMMON_DIR)/common.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/common.cpp -o obj/common.o

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	$(CXX) $(CXXFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

