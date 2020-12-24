CC 			= g++ -std=c++11
CFLAGS		= -c -Wall
LFLAGS 		= -Wall

IMG_DIR		= ./ryuimg
IMG_CPP		= $(wildcard $(IMG_DIR)/*.cpp)
IMG_O		= $(patsubst %.cpp, %.o, $(IMG_CPP))



ryuocr: main.cpp $(IMG_O)
	$(CC) $(LFLAGS) $^ -o $@


$(IMG_O):%.o:%.cpp
	$(CC) $(CFLAGS) $< -o $@

	

.PHONY : clean testpath
clean:
	rm -rf ryuimg/*.o *.o ryuocr a.out

testpath:
	@echo $(IMG_O)