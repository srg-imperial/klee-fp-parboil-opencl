
BENCHMARK=cp
VERSION=cuda
COMMON_DIR=../../../../common
CFLAGS=

all: obj/common.o
	nvcc $(CFLAGS) --use_fast_math -I$(COMMON_DIR) cuenergy_pre8_coalesce.cu main.cu obj/common.o -o ../../bin/$(BENCHMARK)-$(VERSION)
	#nvcc $(CFLAGS) --use_fast_math -I$(COMMON_DIR) cuenergy_pre.cu main.cu obj/common.o -o ../../bin/$(BENCHMARK)-$(VERSION)

obj/common.o: $(COMMON_DIR)/common.cpp $(COMMON_DIR)/common.h
	g++ $(CFLAGS) -c $(COMMON_DIR)/common.cpp -o obj/common.o

clean:
	rm obj/*


