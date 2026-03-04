NASM      = nasm
NFLAGS    = -f elf64 -g -F dwarf -I include/
LD        = ld
LDFLAGS   =

SRCDIR    = src
INCDIR    = include
BUILDDIR  = build

SRCS      = $(shell find $(SRCDIR) -name '*.asm')
OBJS      = $(patsubst $(SRCDIR)/%.asm,$(BUILDDIR)/%.o,$(SRCS))

TARGET    = $(BUILDDIR)/nasmlearn

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.asm $(wildcard $(INCDIR)/*.inc)
	@mkdir -p $(dir $@)
	$(NASM) $(NFLAGS) -o $@ $<

clean:
	rm -rf $(BUILDDIR)
