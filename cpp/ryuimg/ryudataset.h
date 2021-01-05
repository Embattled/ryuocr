#pragma once
#ifndef _RYUDATASET_H
#define _RYUDATASET_H

namespace ryu{

    class mat;
    class dataset{
    private:

        int num;
        int *labels;
        mat* mats;

        // allocate memory
        int initialize(int num);
        
        // Idic
        int readIdicData(std::FILE* fp);
    public:
        dataset();
        ~dataset();
        void clear();
        void printInfo();
        void printInfoHead();

        // read idicDataset
        int loadIdic(const char* filename);


        // write all image data to save dataset
        // int write();
        // int save(const char* filename);
    };
}

#endif