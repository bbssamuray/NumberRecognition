
OBJS = main.o neurolib.o
CC = g++
COMPILER_FLAGS = -O2
LINKER_FLAGS =
OBJ_NAME = numberRecognition.exe

.PHONY:all
all:executable

executable: $(OBJS)
	$(CC) $^ $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $(OBJ_NAME)

%.o : %.c
	$(CC) $(CFLAGS) -c $<

.PHONY:clean
clean:
	-rm -f ./*.o ./*.out ./*.exe
