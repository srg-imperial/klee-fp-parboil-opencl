
include ../../../../common/common.mk

BENCHMARK=mri-fhd
VERSION=opencl
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/file.o obj/common.o obj/ocl-wrapper.o obj/computeFH.o
	g++ -lOpenCL $^ -o ../../bin/$(BENCHMARK)-$(VERSION)
	cp computeFH.cl ../../bin/

obj/main.o: main.cpp
	g++ $(CFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c main.cpp -o obj/main.o

obj/computeFH.o: computeFH.cpp
	g++ $(CFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c computeFH.cpp -o obj/computeFH.o

obj/file.o: file.cpp file.h
	g++ $(CFLAGS) -I$(COMMON_DIR) -c file.cpp -o obj/file.o

obj/common.o: $(COMMON_DIR)/common.cpp $(COMMON_DIR)/common.h
	g++ $(CFLAGS) -c $(COMMON_DIR)/common.cpp -o obj/common.o

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	g++ $(CFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

