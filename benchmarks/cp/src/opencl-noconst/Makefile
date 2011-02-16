
include ../../../../common/common.mk

BENCHMARK=cp
VERSION=opencl-noconst
COMMON_DIR=../../../../common
CLFAGS=

all: obj/main.o obj/common.o obj/ocl-wrapper.o
	g++ -lOpenCL obj/common.o obj/ocl-wrapper.o obj/main.o -o ../../bin/$(BENCHMARK)-$(VERSION)
	cp cuenergy_pre8_coalesce-noconst.cl ../../bin/

obj/main.o: main.cpp cuenergy.h
	g++ $(CFLAGS) -I$(COMMON_DIR) -D OPENCL_PLATFORM="\"$(OPENCL_PLATFORM)\"" -D OPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -c main.cpp -o obj/main.o

obj/common.o: $(COMMON_DIR)/common.cpp $(COMMON_DIR)/common.h
	g++ $(CFLAGS) -c $(COMMON_DIR)/common.cpp -o obj/common.o

obj/ocl-wrapper.o: $(COMMON_DIR)/ocl-wrapper.cpp $(COMMON_DIR)/ocl-wrapper.h
	g++ $(CFLAGS) -c $(COMMON_DIR)/ocl-wrapper.cpp -o obj/ocl-wrapper.o

clean:
	rm -f obj/*

