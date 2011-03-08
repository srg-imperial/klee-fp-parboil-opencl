CXXLD = g++
CFLAGS += -I../../../../common -std=gnu99
CXXFLAGS += -I../../../../common

all: mri-fhd

OBJS = main.o file.o computeFH.o ../../../../common/parboil.o

mri-fhd: $(OBJS)
	$(CXXLD) -o $@ $^

clean:
	rm -f $(OBJS) mri-q
