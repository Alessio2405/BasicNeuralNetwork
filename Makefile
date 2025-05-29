CC      := gcc
CFLAGS  := -std=c99 -Wall -Wextra -pedantic -Iinclude
LDFLAGS := -lm

SRCDIR  := src
OBJDIR  := build
BINDIR  := bin

TARGET  := nn

tSRCS    := $(wildcard $(SRCDIR)/*.c)
OBJS    := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))

all: dirs $(BINDIR)/$(TARGET)

dirs:
	@mkdir -p $(OBJDIR) $(BINDIR)

	\$(CC) \$(CFLAGS) -c $< -o $@

$(BINDIR)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all dirs clean