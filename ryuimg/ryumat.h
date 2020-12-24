#pragma once
#ifndef _RYUMAT_H
#define _RYUMAT_H

namespace ryu
{
    class mat
    {
    private:
        int width;
        int height; 
        int channel;
        unsigned char *data;

        int linesize;
        int size;
    public:
        mat();
        mat(int width, int height, int channel);
        ~mat();

        void clear();
        int initialize();
        int initialize(int width, int height, int channel);


        // Useful
        int getSum();
        int subAll(unsigned char x);

        // get parameter functions
        int getWidth() const { return width; }
        int getheight() const { return height; }
        int getChannel() const { return channel; }
        unsigned char * getData() const { return data; };
        int getLineSize() const { return linesize; };
        int getSize() const { return size; };

        // operator
        bool operator==(const mat &mat2);
        void operator=(const mat &mat2);

        //io
        int readFromIdic(std::FILE* fp);

        // print
        void printInfo();

    };
    
} // namespace ryu


#endif