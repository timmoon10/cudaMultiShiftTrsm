CC = nvcc
EXE = validation
CFLAGS   = -Xcompiler -fopenmp -Xcompiler -Wall
INCFLAGS = -Iinclude
LIBFLAGS = -lm -lblas -lcublas -llapack

SOURCES     = $(wildcard src/*.cu)
INCLUDES    = $(wildcard include/%.hpp)
OBJECTS     = $(patsubst src/%.cu,obj/%.o,$(SOURCES))
EXESOURCES  = $(wildcard *.cu)
EXECUTABLES = $(patsubst %.cu,%,$(EXESOURCES))

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
