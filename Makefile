# Makefile for Grapheus project
# Requires CUDA toolkit and a C++17 compiler

# Compiler options
NVCC = nvcc
CXXFLAGS = -fopenmp -march=native
NVCCFLAGS = -use_fast_math -O3 -DNDEBUG -std=c++20

# Combine CXXFLAGS into NVCCFLAGS
NVCCFLAGS += $(addprefix --compiler-options ,$(CXXFLAGS))

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
all: $(EXE)

$(EXE): $(OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ $(LIBS) -o $(EXE)

$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)
