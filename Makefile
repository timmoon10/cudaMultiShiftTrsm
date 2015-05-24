# Compiler parameters
CC = nvcc
CFLAGS   = -Xcompiler -fopenmp -Xcompiler -Wall
INCFLAGS = -Iinclude
LIBFLAGS = -lm -lblas -lcublas -llapack

# Files
EXE = validation
SOURCES     = $(wildcard src/*.cu)
INCLUDES    = $(wildcard include/*.hpp)
OBJECTS     = $(patsubst src/%.cu,obj/%.o,$(SOURCES))
EXESOURCES  = $(wildcard *.cu)
EXECUTABLES = $(patsubst %.cu,%,$(EXESOURCES))

# Command line options
#   DEBUG=1      Activate debugging flag
#   DATAFLOAT=#  Choose data type associated with datafloat
#   O=1          Activate compiler optimization flag

# Debug option
ifeq ($(DATAFLOAT),float)
CFLAGS += -DDATAFLOAT_DEFINED -Ddatafloat=float
endif
ifeq ($(DATAFLOAT),double)
CFLAGS += -DDATAFLOAT_DEFINED -Ddatafloat=double
endif
ifeq ($(DATAFLOAT),complex<float>)
CFLAGS += -DDATAFLOAT_DEFINED -Ddatafloat="complex<float>"
endif
ifeq ($(DATAFLOAT),complex<double>)
CFLAGS += -DDATAFLOAT_DEFINED -Ddatafloat="complex<double>"
endif

# Debug option
ifeq ($(DEBUG),1)
CFLAGS += -g -pg
endif

# Optimization option
ifeq ($(O),1)
CFLAGS += -O2
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
