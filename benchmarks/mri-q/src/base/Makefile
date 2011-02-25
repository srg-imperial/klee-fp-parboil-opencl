CXXLD = g++
CFLAGS += -I../../../../common -std=gnu99

all: mri-q

OBJS = main.o file.o computeQ.o ../../../../common/parboil.o

mri-q: $(OBJS)
	$(CXXLD) -o $@ $^

clean:
	rm -f $(OBJS) mri-q
