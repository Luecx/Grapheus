# Makefile for Grapheus project
# Requires CUDA toolkit and a C++17 compiler

# Compiler options
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -stdlib=libc++
NVCCFLAGS = -use_fast_math -O3 -DNDEBUG -std=c++17 -Xcompiler -fopenmp

# Libraries
LIBS = -lcublas

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Files
SRCS := $(sort $(shell find $(SRCDIR) -name '*.cu'))
OBJS := $(SRCS:$(SRCDIR)/%.cu=$(OBJDIR)/%.obj)
EXE  := $(BINDIR)/Grapheus

# Targets
build: $(EXE)

$(EXE): $(OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ $(LIBS) -o $(EXE)

$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

util: NVCCFLAGS += -DUTILITIES
util: $(OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ $(LIBS) -o $(EXE)

clean:
	rm -rf $(OBJDIR) $(BINDIR)
