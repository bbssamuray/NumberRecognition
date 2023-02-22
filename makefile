
OBJS = main.o neurolib.o
CC = g++
CXXFLAGS = -O2
LDFLAGS =
OBJ_NAME = numberRecognition.exe

.PHONY:all
all:executable

executable: $(OBJS)
	$(CC) $^ $(CXXFLAGS) $(LDFLAGS) -o $(OBJ_NAME)

#This doesn't even get invoked because of the implicit rules smh
%.o : %.c
	$(CC) $(CFLAGS) -c $<

.PHONY:clean
clean:
	-rm -f ./*.o ./*.out ./*.exe
