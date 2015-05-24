# Compiler parameters
CC = nvcc
CFLAGS   = -Xcompiler -fopenmp -Xcompiler -Wall
INCFLAGS = -Iinclude
LIBFLAGS = -lm -lblas -lcublas -llapack

# Files
SOURCES     = $(wildcard src/*.cu)
INCLUDES    = $(wildcard include/*.hpp)
OBJECTS     = $(patsubst src/%.cu,obj/%.o,$(SOURCES))
EXESOURCES  = $(wildcard *.cu)
EXECUTABLES = $(patsubst %.cu,%,$(EXESOURCES))

# Command line options
#   DEBUG=1      Activate debugging flag
#   O=#          Activate compiler optimization flag

# Debug option
ifeq ($(DEBUG),1)
CFLAGS += -g -pg
endif

# Optimization option
ifeq ($(O),0)
CFLAGS += -O0
endif
ifeq ($(O),1)
CFLAGS += -O1
endif
ifeq ($(O),2)
CFLAGS += -O2
endif
ifeq ($(O),0)
CFLAGS += -O3
endif

# Create all executables
default: $(EXECUTABLES)

# Compile object file
obj/%.o: src/%.cu $(INCLUDES)
	$(CC) -c $(CFLAGS) $< $(INCFLAGS) -o $@

# Link executable
%: %.cu $(OBJECTS) $(INCLUDES)
	$(CC) $< $(OBJECTS) $(CFLAGS) $(INCFLAGS) $(LIBFLAGS) -o $@

# Prevents deleting object files
.SECONDARY: $(OBJECTS)

# Clean directory
clean:
	rm -f $(EXECUTABLES) obj/* *.o
