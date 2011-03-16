CXXLD = g++
CFLAGS += -I../../../../common -std=gnu99
CXXFLAGS += -I../../../../common

all: cp

OBJS = main.o cenergy.o ../../../../common/parboil.o

cp: $(OBJS)
	$(CXXLD) -o $@ $^

clean:
	rm -f $(OBJS) cp
